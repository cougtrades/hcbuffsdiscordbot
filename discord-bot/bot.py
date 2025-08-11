import os
import re
import json
import base64
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
import requests

# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Environment and clients
# --------------------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAIKEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not OPENAI_API_KEY:
    logger.error("OPENAIKEY not found in environment")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Using a constant for model selection improves maintainability
OPENAI_MODEL_NAME = "gpt-4.1"

# --------------------------------------------------------------------------------------
# Discord bot setup
# --------------------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Configuration for the buff channels
BUFF_CHANNEL: Dict[str, tuple[int, str]] = {
    "horde": (1306953557378859119, "horde"),
    "alliance": (1336750424048271381, "alliance"),
    "both": (1366901374826315837, "both"),
}

# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------
@dataclass
class BuffEntry:
    datetime: str
    guild: str
    buff: str
    notes: str
    server: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BuffEntry":
        return BuffEntry(
            datetime=str(data["datetime"]).strip(),
            guild=str(data["guild"]).strip(),
            buff=str(data["buff"]).strip(),
            notes=str(data.get("notes", "")).strip(),
            server=str(data.get("server", "Doomhowl")).strip() or "Doomhowl",
        )

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def clean_json_from_text(text: str) -> str:
    """Extract a JSON array from a text that may contain markdown fences or prose.
    Returns the raw JSON string representing an array.
    """
    cleaned = text.strip()
    # Remove common markdown fences
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    # Try to extract the outermost JSON array if extra text is present
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def is_duplicate_buff(new_buff: BuffEntry, existing: List[Dict[str, Any]]) -> bool:
    for existing_buff in existing:
        if (
            existing_buff.get("datetime") == new_buff.datetime
            and str(existing_buff.get("guild", "")).lower() == new_buff.guild.lower()
            and existing_buff.get("buff") == new_buff.buff
        ):
            return True
    return False


def detect_faction_from_message(message_content: str, channel_type: Optional[str] = None) -> Optional[str]:
    """Detect faction from message content based on channel type and emojis/keywords."""
    if channel_type == "both":
        lower = message_content.lower()
        # Support both custom emoji names and plain words
        if ":horde:" in lower or "horde" in lower:
            return "horde"
        if ":alliance:" in lower or "alliance" in lower:
            return "alliance"
        return None
    return channel_type


def contains_cancel_intent(message_content: str) -> bool:
    """Heuristically detect if a message intends to cancel/delete a prior buff announcement."""
    lower = message_content.lower()
    # Simple, conservative keyword set with word boundaries
    cancel_patterns = [
        r"\bcancel\b",
        r"\bcanceled\b",
        r"\bcancelled\b",
        r"\bdelete\b",
        r"\bremoved?\b",
        r"\bvoid\b",
        r"\bnevermind\b",
        r"\bnm\b",
        r"\bignore\b",
        r"\bwrong\b",
        r"\bcall it off\b",
    ]
    return any(re.search(pat, lower) for pat in cancel_patterns)


# --------------------------------------------------------------------------------------
# OpenAI parsing (blocking + async wrapper)
# --------------------------------------------------------------------------------------

def parse_buff_with_ai_blocking(message_content: str) -> List[Dict[str, Any]]:
    """Blocking call to OpenAI API to parse a buff message into structured data.
    Wrapped by an async adapter to avoid blocking the event loop.
    """
    if client is None:
        raise RuntimeError("OpenAI client is not configured")

    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are an expert data extraction assistant tasked with parsing Discord messages about World of Warcraft buffs. Your goal is to extract structured information while being flexible enough to handle various human-written formats.

Key Requirements:
1. Extract these fields from the message:
   - Guild Name: Usually in <> tags
   - Buff Type: One of: Zandalar, Nefarian, Rend, or Onyxia
   - Date: Any date format (e.g., "Tuesday June 10th", "6/10", etc.)
   - Time: Any time format with "ST" (Server Time)
   - Notes: Any additional information in parentheses or after the time

2. Time Conversion Rules (CRITICAL):
   - Server Time (ST) is Mountain Time (America/Denver)
   - Convert all times to UTC/Z format
   - For times 18:00-23:59 MT: Add 6 hours and use next day's date
   - For times 00:00-17:59 MT: Add 6 hours and use same day's date
   - If no date is specified in the message, use today's date ({today})
   - Always use server name "Doomhowl"
   - Handle DST: After March 9, 2025, use 6-hour offset; before that, use 7-hour offset
   - IMPORTANT: Both Horde and Alliance use the same time conversion rules
   - Example: 6:55 PM ST becomes 00:55 UTC next day (not 01:55)

3. Buff Name Standardization:
   - Zandalar: Accepts ZG, Zul'Gurub, etc.
   - Nefarian: Accepts Nef, Nefarian, etc.
   - Rend: Accepts Rend, Warchief Rend Blackhand, etc.
   - Onyxia: Accepts Ony, Onyxia, etc.

4. Time Format Handling:
   - Handle both 12-hour and 24-hour formats
   - Handle various separators (:, ., space)
   - Handle various AM/PM indicators or lack thereof
   - Handle various date formats
   - Handle various timezone indicators (ST, Server Time, etc.)
   - When no date is specified, use today's date ({today})

Process this message and return the extracted information in JSON format:
{message_content}

Return the data in this exact format:
[
    {{
        "datetime": "2025-MM-DDTHH:MM:00Z",
        "guild": "Guild Name",
        "buff": "Zandalar|Nefarian|Rend|Onyxia",
        "notes": "",
        "server": "Doomhowl"
    }}
]"""

    system_prompt = f"""You are a WoW buff message parser that handles various human-written formats.
Your task is to extract buff information and convert times to UTC/Z format.
Key rules:
1. Server Time (ST) is Mountain Time
2. Convert all times to UTC/Z by adding 6 hours
3. Use next day's date for times 18:00-23:59 MT
4. Use same day's date for times 00:00-17:59 MT
5. If no date is specified, use today's date ({today})
6. Always use server name Doomhowl
7. Handle DST: After March 9, 2025, use 6-hour offset; before that, use 7-hour offset
8. Standardize buff names to: Zandalar, Nefarian, Rend, or Onyxia
9. Be flexible with time formats - handle both 12/24 hour, various separators, and AM/PM indicators
10. Both Horde and Alliance use the same time conversion rules
11. Example: 6:55 PM ST becomes 00:55 UTC next day (not 01:55)"""

    # Simple retry with exponential backoff
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = response.choices[0].message.content or "[]"
            cleaned = clean_json_from_text(raw)
            data = json.loads(cleaned)
            if not isinstance(data, list):
                logger.warning("AI parse did not return a list; coercing to list if possible")
                data = [data]
            return data
        except Exception as err:  # noqa: BLE001
            last_err = err
            sleep_seconds = 1.5 * (2 ** attempt)
            logger.warning("AI parse attempt %s failed: %s. Retrying in %.1fs", attempt + 1, err, sleep_seconds)
            time_to_sleep = min(sleep_seconds, 6)
            # Sleep inside blocking function
            import time as _time  # local import to avoid top-level shadowing

            _time.sleep(time_to_sleep)
    # If all retries failed, raise a consolidated error
    raise RuntimeError(f"Failed to parse message with AI after retries: {last_err}")


async def parse_buff_with_ai(message_content: str) -> Optional[List[Dict[str, Any]]]:
    try:
        return await asyncio.to_thread(parse_buff_with_ai_blocking, message_content)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error parsing with AI: %s", exc)
        return None

# --------------------------------------------------------------------------------------
# GitHub publishing (blocking + async wrapper)
# --------------------------------------------------------------------------------------

def publish_buff_to_github_blocking(
    buff_data: Dict[str, Any],
    *,
    is_edit: bool = False,
    original_message: Optional[str] = None,
    delete_only: bool = False,
) -> bool:
    if not GITHUB_TOKEN:
        logger.error("GitHub token not configured")
        return False

    faction = buff_data.get("faction")
    if faction not in {"horde", "alliance"}:
        logger.error("Invalid faction: %s", faction)
        return False

    path = f"{faction}_buffs.json"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Use a short-lived Session per call to benefit from connection pooling
    with requests.Session() as session:
        session.headers.update(headers)

        # Fetch current file
        url = f"https://api.github.com/repos/cougtrades/wowbuffs/contents/{path}"
        file_response = session.get(url, timeout=20)
        if not file_response.ok:
            logger.error("Failed to fetch file %s: %s", path, file_response.text)
            return False

        file_data = file_response.json()
        current_content: List[Dict[str, Any]] = json.loads(base64.b64decode(file_data["content"]).decode("utf-8"))

        # Delete-only path: remove based on the original message and update file
        if delete_only:
            if not original_message:
                logger.error("delete_only requested but original_message was not provided")
                return False
            try:
                original_list = parse_buff_with_ai_blocking(original_message)
                if original_list:
                    original_buff = BuffEntry.from_dict(original_list[0])
                    before_len = len(current_content)
                    current_content = [
                        b
                        for b in current_content
                        if not (
                            b.get("datetime") == original_buff.datetime
                            and str(b.get("guild", "")).lower() == original_buff.guild.lower()
                            and b.get("buff") == original_buff.buff
                        )
                    ]
                    if len(current_content) == before_len:
                        logger.info("No matching buff to remove for delete request")
                else:
                    logger.info("No structured buff found in original message for deletion")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse original message during delete: %s", exc)

            update_response = session.put(
                url,
                json={
                    "message": f'Remove entry from {path} due to cancellation/deletion',
                    "content": base64.b64encode(json.dumps(current_content, indent=4).encode("utf-8")).decode("utf-8"),
                    "sha": file_data["sha"],
                },
                timeout=20,
            )

            if not update_response.ok:
                logger.error("Failed to update file %s: %s", path, update_response.text)
                return False

            logger.info("Removed buff entry on request")
            return True

        new_buff = BuffEntry.from_dict(
            {
                "datetime": buff_data["datetime"],
                "guild": buff_data["guild"],
                "buff": buff_data["buff"],
                "notes": buff_data.get("notes", ""),
                "server": "Doomhowl",
            }
        )

        # Duplicate check
        if is_duplicate_buff(new_buff, current_content):
            logger.info(
                "Skipping duplicate buff: %s %s at %s",
                new_buff.guild,
                new_buff.buff,
                new_buff.datetime,
            )
            return True

        # If this is an edit, parse the original message and remove the old entry
        if is_edit and original_message:
            try:
                original_list = parse_buff_with_ai_blocking(original_message)
                if original_list:
                    original_buff = BuffEntry.from_dict(original_list[0])
                    current_content = [
                        b
                        for b in current_content
                        if not (
                            b.get("datetime") == original_buff.datetime
                            and str(b.get("guild", "")).lower() == original_buff.guild.lower()
                            and b.get("buff") == original_buff.buff
                        )
                    ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse original message during edit cleanup: %s", exc)

        # Add new buff and sort
        current_content.append(new_buff.__dict__)
        current_content.sort(key=lambda x: x["datetime"])  # ISO datetime sorts lexicographically

        # Update file on GitHub
        update_response = session.put(
            url,
            json={
                "message": f'Update {path} with new buff from Discord: {new_buff.guild} {new_buff.buff}',
                "content": base64.b64encode(json.dumps(current_content, indent=4).encode("utf-8")).decode("utf-8"),
                "sha": file_data["sha"],
            },
            timeout=20,
        )

        if not update_response.ok:
            logger.error("Failed to update file %s: %s", path, update_response.text)
            return False

        logger.info("Published buff: %s %s", new_buff.guild, new_buff.buff)
        return True


async def publish_buff_to_github(
    buff_data: Dict[str, Any],
    *,
    is_edit: bool = False,
    original_message: Optional[str] = None,
    delete_only: bool = False,
) -> bool:
    try:
        return await asyncio.to_thread(
            publish_buff_to_github_blocking,
            buff_data,
            is_edit=is_edit,
            original_message=original_message,
            delete_only=delete_only,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error publishing to GitHub: %s", exc)
        return False

# --------------------------------------------------------------------------------------
# Centralized message handling
# --------------------------------------------------------------------------------------

async def process_and_publish(
    message_content: str,
    detected_faction: Optional[str],
    *,
    is_edit: bool = False,
    original_message: Optional[str] = None,
    author_for_logs: str = "",
    original_message_for_parse: Optional[str] = None,
) -> None:
    if not detected_faction:
        logger.info("Could not detect faction from message content")
        return

    logger.info("Processing message from %s", author_for_logs or "unknown")
    logger.debug("Content: %s", message_content)

    # If we have original context (reply/edit), combine it to aid parsing updates with partial info
    parse_input = message_content
    if original_message_for_parse:
        parse_input = (
            f"Original announcement:\n{original_message_for_parse}\n\n"
            f"Update message:\n{message_content}\n\n"
            f"Extract and return only the final, updated buff announcement."
        )

    buff_data_list = await parse_buff_with_ai(parse_input)
    if not buff_data_list:
        logger.info("No buffs found in the message or parsing failed")
        return

    logger.info("Found %d buff(s)", len(buff_data_list))
    for buff_data in buff_data_list:
        # Validate and enrich
        try:
            entry = BuffEntry.from_dict(buff_data)
        except KeyError as exc:  # missing keys
            logger.warning("Skipping malformed buff entry missing %s: %s", exc, buff_data)
            continue

        logger.info("- %s by %s at %s", entry.buff, entry.guild, entry.datetime)
        buff_payload: Dict[str, Any] = {
            "datetime": entry.datetime,
            "guild": entry.guild,
            "buff": entry.buff,
            "notes": entry.notes,
            "server": entry.server,
            "faction": detected_faction,
        }
        await publish_buff_to_github(
            buff_payload,
            is_edit=is_edit,
            original_message=original_message,
        )

# --------------------------------------------------------------------------------------
# Discord events and commands
# --------------------------------------------------------------------------------------

@bot.event
async def on_ready() -> None:
    logger.info("Bot is ready! Logged in as %s", bot.user.name if bot.user else "<unknown>")

    # Monitor all configured channels by fetching the last message per channel
    for faction_key, (channel_id, channel_type) in BUFF_CHANNEL.items():
        channel = bot.get_channel(channel_id)
        if not channel:
            logger.error("Channel not found (ID: %s)", channel_id)
            continue

        logger.info("Monitoring %s channel: %s", faction_key, getattr(channel, "name", str(channel_id)))

        try:
            async for message in channel.history(limit=1):
                if message.author == bot.user:
                    continue
                detected_faction = detect_faction_from_message(message.content, channel_type)
                await process_and_publish(
                    message.content,
                    detected_faction,
                    author_for_logs=getattr(message.author, "name", ""),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read history for channel %s: %s", channel_id, exc)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author == bot.user:
        return

    # Check if message is from any of our buff channels
    channel_faction: Optional[str] = None
    channel_type: Optional[str] = None
    for faction_key, (channel_id, ct) in BUFF_CHANNEL.items():
        if message.channel.id == channel_id:
            channel_type = ct
            channel_faction = detect_faction_from_message(message.content, ct)
            break

    if not channel_type:
        await bot.process_commands(message)
        return

    # If the message is a reply to an earlier announcement, treat it as an update/cancel
    if message.reference and message.reference.message_id:
        referenced_message: Optional[discord.Message] = None
        try:
            # Prefer resolved reference when available
            if message.reference.resolved and isinstance(message.reference.resolved, discord.Message):
                referenced_message = message.reference.resolved
            else:
                referenced_message = await message.channel.fetch_message(message.reference.message_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch referenced message: %s", exc)
            referenced_message = None

        if referenced_message:
            # Determine faction based on the referenced content if channel is 'both'
            ref_faction = channel_faction if channel_type != "both" else detect_faction_from_message(referenced_message.content, "both")

            # Handle cancellation intent
            if contains_cancel_intent(message.content):
                factions_to_process: List[str] = []
                if ref_faction in {"horde", "alliance"}:
                    factions_to_process = [ref_faction]
                else:
                    factions_to_process = ["horde", "alliance"]

                for fac in factions_to_process:
                    await publish_buff_to_github(
                        {"faction": fac},
                        is_edit=False,
                        original_message=referenced_message.content,
                        delete_only=True,
                    )
                await bot.process_commands(message)
                return

            # Treat as an edit/update (use original content for both parsing context and cleanup)
            detected_faction_effective = ref_faction if ref_faction else channel_faction
            await process_and_publish(
                message.content,
                detected_faction_effective,
                is_edit=True,
                original_message=referenced_message.content,
                author_for_logs=getattr(message.author, "name", ""),
                original_message_for_parse=referenced_message.content,
            )
            await bot.process_commands(message)
            return

    # Non-reply fallback: treat as a fresh announcement if faction is detectable
    if not channel_faction:
        await bot.process_commands(message)
        return

    await process_and_publish(
        message.content,
        channel_faction,
        author_for_logs=getattr(message.author, "name", ""),
    )

    await bot.process_commands(message)


@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message) -> None:
    if after.author == bot.user:
        return

    # Check if message is from any of our buff channels
    channel_faction: Optional[str] = None
    channel_type: Optional[str] = None
    for faction_key, (channel_id, ct) in BUFF_CHANNEL.items():
        if after.channel.id == channel_id:
            channel_type = ct
            channel_faction = detect_faction_from_message(after.content, ct)
            break

    if not channel_type:
        return

    # If channel is 'both', try to infer faction from the previous content when missing
    effective_faction = channel_faction
    if channel_type == "both" and not effective_faction:
        effective_faction = detect_faction_from_message(before.content, "both")

    await process_and_publish(
        after.content,
        effective_faction,
        is_edit=True,
        original_message=before.content,
        author_for_logs=getattr(after.author, "name", ""),
        original_message_for_parse=before.content,
    )


@bot.event
async def on_message_delete(message: discord.Message) -> None:
    # Remove the corresponding buff entry when a message is deleted in our channels
    if message.author == bot.user:
        return

    channel_type: Optional[str] = None
    for faction_key, (channel_id, ct) in BUFF_CHANNEL.items():
        if message.channel.id == channel_id:
            channel_type = ct
            break

    if not channel_type:
        return

    # Determine faction(s) to remove from
    factions_to_process: List[str] = []
    detected = detect_faction_from_message(message.content, channel_type)
    if detected in {"horde", "alliance"}:
        factions_to_process = [detected]
    elif channel_type == "both":
        factions_to_process = ["horde", "alliance"]

    for fac in factions_to_process:
        await publish_buff_to_github(
            {"faction": fac},
            is_edit=False,
            original_message=message.content,
            delete_only=True,
        )


@bot.command(name="test_parse")
async def test_parse(ctx: commands.Context, *, message_content: str) -> None:
    """Test the message parser with custom content."""
    buff_data_list = await parse_buff_with_ai(message_content)
    if not buff_data_list:
        await ctx.send("No buffs found in the message.")
        return

    lines = [f"Found {len(buff_data_list)} buffs:"]
    for i, buff_data in enumerate(buff_data_list, start=1):
        try:
            entry = BuffEntry.from_dict(buff_data)
        except KeyError:
            continue
        lines.append(f"{i}. **{entry.buff}** by *{entry.guild}* at `{entry.datetime}`")
    await ctx.send("\n".join(lines))

# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------

def main() -> None:
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN not found in environment")
        print("Error: DISCORD_TOKEN not found in .env file")
        return
    if not OPENAI_API_KEY:
        logger.error("OPENAIKEY not found in environment")
        print("Error: OPENAIKEY not found in .env file")
        return
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not found in environment")
        print("Error: GITHUB_TOKEN not found in .env file")
        return

    try:
        bot.run(DISCORD_TOKEN)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error running bot: %s", exc)
        print(f"Error running bot: {exc}")


if __name__ == "__main__":
    main()