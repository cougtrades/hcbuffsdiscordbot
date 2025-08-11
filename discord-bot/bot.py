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

# In-memory cache mapping Discord message ID -> list of published BuffEntry
MESSAGE_ID_TO_BUFFS: Dict[int, List[BuffEntry]] = {}

# Keywords that indicate a cancellation/removal in a reply
CANCEL_KEYWORDS = {
    "cancel", "canceled", "cancelled", "delete", "remove", "revoke", "scratch", "no go", "nogo", "abort",
}


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

# --------------------------------------------------------------------------------------
# OpenAI parsing (blocking + async wrapper)
# --------------------------------------------------------------------------------------

def parse_buff_with_ai_blocking(message_content: str, anchor_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Blocking call to OpenAI API to parse a buff message into structured data.
    Wrapped by an async adapter to avoid blocking the event loop.

    anchor_date: if provided (YYYY-MM-DD), use this as "today" for prompts where date is missing
    to ensure edits/deletes use the original message's date context.
    """
    if client is None:
        raise RuntimeError("OpenAI client is not configured")

    today = anchor_date or datetime.now().strftime("%Y-%m-%d")

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


async def parse_buff_with_ai(message_content: str, *, anchor_date: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    try:
        return await asyncio.to_thread(parse_buff_with_ai_blocking, message_content, anchor_date)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error parsing with AI: %s", exc)
        return None

# --------------------------------------------------------------------------------------
# GitHub publishing (blocking + async wrapper)
# --------------------------------------------------------------------------------------

def _github_fetch_current(session: requests.Session, url: str) -> Optional[Dict[str, Any]]:
    resp = session.get(url, timeout=20)
    if not resp.ok:
        logger.error("Failed to fetch file: %s", resp.text)
        return None
    return resp.json()


def _github_put_with_retry(session: requests.Session, url: str, content_list: List[Dict[str, Any]], sha: str, commit_message: str) -> bool:
    body = {
        "message": commit_message,
        "content": base64.b64encode(json.dumps(content_list, indent=4).encode("utf-8")).decode("utf-8"),
        "sha": sha,
    }
    # Retry a few times on 409 conflicts
    for attempt in range(3):
        put_resp = session.put(url, json=body, timeout=20)
        if put_resp.ok:
            return True
        if put_resp.status_code == 409:
            # Refetch latest sha and retry once
            latest = _github_fetch_current(session, url)
            if not latest:
                return False
            body["sha"] = latest.get("sha", sha)
            continue
        logger.error("Failed to update file: %s", put_resp.text)
        return False
    return False


def publish_buff_to_github_blocking(buff_data: Dict[str, Any], *, is_edit: bool = False, original_message: Optional[str] = None, old_entries_to_remove: Optional[List[BuffEntry]] = None, anchor_date_for_original: Optional[str] = None) -> bool:
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

    with requests.Session() as session:
        session.headers.update(headers)
        url = f"https://api.github.com/repos/cougtrades/wowbuffs/contents/{path}"

        file_data = _github_fetch_current(session, url)
        if not file_data:
            return False

        current_content: List[Dict[str, Any]] = json.loads(base64.b64decode(file_data["content"]).decode("utf-8"))

        # Remove old entries if provided (for edit or replacements)
        if old_entries_to_remove:
            remove_keys = {(e.datetime, e.guild.lower(), e.buff) for e in old_entries_to_remove}
            current_content = [
                b for b in current_content
                if (b.get("datetime"), str(b.get("guild", "")).lower(), b.get("buff")) not in remove_keys
            ]

        # Fallback: if is_edit and no explicit entries, try parse original_message anchored to its date
        elif is_edit and original_message:
            try:
                original_list = parse_buff_with_ai_blocking(original_message, anchor_date=anchor_date_for_original)
                if original_list:
                    original_buff = BuffEntry.from_dict(original_list[0])
                    current_content = [
                        b for b in current_content
                        if not (
                            b.get("datetime") == original_buff.datetime
                            and str(b.get("guild", "")).lower() == original_buff.guild.lower()
                            and b.get("buff") == original_buff.buff
                        )
                    ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse original message during edit cleanup: %s", exc)

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

        # Add new buff and sort
        current_content.append(new_buff.__dict__)
        current_content.sort(key=lambda x: x["datetime"])  # ISO datetime sorts lexicographically

        ok = _github_put_with_retry(
            session,
            url,
            current_content,
            file_data.get("sha", ""),
            f'Update {path} with new buff from Discord: {new_buff.guild} {new_buff.buff}',
        )
        if not ok:
            return False

        logger.info("Published buff: %s %s", new_buff.guild, new_buff.buff)
        return True


async def publish_buff_to_github(buff_data: Dict[str, Any], *, is_edit: bool = False, original_message: Optional[str] = None, old_entries_to_remove: Optional[List[BuffEntry]] = None, anchor_date_for_original: Optional[str] = None) -> bool:
    try:
        return await asyncio.to_thread(
            publish_buff_to_github_blocking,
            buff_data,
            is_edit=is_edit,
            original_message=original_message,
            old_entries_to_remove=old_entries_to_remove,
            anchor_date_for_original=anchor_date_for_original,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error publishing to GitHub: %s", exc)
        return False


def remove_buffs_from_github_blocking(faction: str, entries_to_remove: List[BuffEntry]) -> bool:
    if not GITHUB_TOKEN:
        logger.error("GitHub token not configured")
        return False
    if faction not in {"horde", "alliance"}:
        logger.error("Invalid faction: %s", faction)
        return False

    path = f"{faction}_buffs.json"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    with requests.Session() as session:
        session.headers.update(headers)
        url = f"https://api.github.com/repos/cougtrades/wowbuffs/contents/{path}"

        file_data = _github_fetch_current(session, url)
        if not file_data:
            return False

        current_content: List[Dict[str, Any]] = json.loads(base64.b64decode(file_data["content"]).decode("utf-8"))
        remove_keys = {(e.datetime, e.guild.lower(), e.buff) for e in entries_to_remove}
        new_content = [
            b for b in current_content
            if (b.get("datetime"), str(b.get("guild", "")).lower(), b.get("buff")) not in remove_keys
        ]

        if new_content == current_content:
            logger.info("No matching entries to remove for %s", path)
            return True

        ok = _github_put_with_retry(
            session,
            url,
            new_content,
            file_data.get("sha", ""),
            f"Remove {len(entries_to_remove)} buff(s) via Discord reply/delete",
        )
        if not ok:
            return False

        logger.info("Removed %d buff(s) from %s", len(entries_to_remove), path)
        return True


async def remove_buffs_from_github(faction: str, entries_to_remove: List[BuffEntry]) -> bool:
    try:
        return await asyncio.to_thread(remove_buffs_from_github_blocking, faction, entries_to_remove)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error removing buffs from GitHub: %s", exc)
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
    message_id: Optional[int] = None,
    anchor_date_str: Optional[str] = None,
    old_entries_to_remove: Optional[List[BuffEntry]] = None,
) -> None:
    if not detected_faction:
        logger.info("Could not detect faction from message content")
        return

    logger.info("Processing message from %s", author_for_logs or "unknown")
    logger.debug("Content: %s", message_content)

    buff_data_list = await parse_buff_with_ai(message_content, anchor_date=anchor_date_str)
    if not buff_data_list:
        logger.info("No buffs found in the message or parsing failed")
        return

    logger.info("Found %d buff(s)", len(buff_data_list))

    published_entries: List[BuffEntry] = []
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
        ok = await publish_buff_to_github(
            buff_payload,
            is_edit=is_edit,
            original_message=original_message,
            old_entries_to_remove=old_entries_to_remove,
            anchor_date_for_original=anchor_date_str,
        )
        if ok:
            published_entries.append(entry)

    # Update cache mapping for subsequent edits/deletes
    if message_id and published_entries:
        MESSAGE_ID_TO_BUFFS[message_id] = published_entries

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
                # Anchor to the posted date so any implicit dates are stable
                anchor_date_str = message.created_at.strftime("%Y-%m-%d") if message.created_at else None
                await process_and_publish(
                    message.content,
                    detected_faction,
                    author_for_logs=getattr(message.author, "name", ""),
                    message_id=message.id,
                    anchor_date_str=anchor_date_str,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read history for channel %s: %s", channel_id, exc)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author == bot.user:
        return

    # Check if message is from any of our buff channels
    channel_type: Optional[str] = None
    for _, (channel_id, ctype) in BUFF_CHANNEL.items():
        if message.channel.id == channel_id:
            channel_type = ctype
            break

    if not channel_type:
        await bot.process_commands(message)
        return

    # Handle replies for cancel or replacement
    if message.reference and message.reference.message_id:
        try:
            referenced: Optional[discord.Message] = message.reference.resolved  # may be None
            if referenced is None:
                referenced = await message.channel.fetch_message(message.reference.message_id)
        except Exception:  # noqa: BLE001
            referenced = None

        if referenced and referenced.author != bot.user:
            detected_faction = detect_faction_from_message(message.content, channel_type)
            # Cancellation command
            if any(kw in message.content.lower() for kw in CANCEL_KEYWORDS):
                entries = MESSAGE_ID_TO_BUFFS.get(referenced.id)
                if not entries:
                    # Fallback: parse referenced content anchored to its date
                    anchor_date_str = referenced.created_at.strftime("%Y-%m-%d") if referenced.created_at else None
                    parsed = await parse_buff_with_ai(referenced.content, anchor_date=anchor_date_str)
                    if parsed:
                        entries = [BuffEntry.from_dict(p) for p in parsed]
                if entries:
                    faction = detect_faction_from_message(referenced.content, channel_type) or detect_faction_from_message(message.content, channel_type)
                    if faction:
                        await remove_buffs_from_github(faction, entries)
                        MESSAGE_ID_TO_BUFFS.pop(referenced.id, None)
                # Do not proceed to normal processing
                await bot.process_commands(message)
                return

            # Replacement: reply contains a new buff; remove old entries for referenced message id, then publish new
            parsed_reply = await parse_buff_with_ai(message.content)
            if parsed_reply:
                old_entries = MESSAGE_ID_TO_BUFFS.get(referenced.id)
                if not old_entries and referenced:
                    anchor_date_str = referenced.created_at.strftime("%Y-%m-%d") if referenced.created_at else None
                    parsed_old = await parse_buff_with_ai(referenced.content, anchor_date=anchor_date_str)
                    if parsed_old:
                        old_entries = [BuffEntry.from_dict(p) for p in parsed_old]
                # Remove old entries first
                faction = detect_faction_from_message(message.content, channel_type)
                if faction and old_entries:
                    await remove_buffs_from_github(faction, old_entries)
                    MESSAGE_ID_TO_BUFFS.pop(referenced.id, None)
                # Then publish new entries while associating them with the original message id
                anchor_date_str_new = message.created_at.strftime("%Y-%m-%d") if message.created_at else None
                await process_and_publish(
                    message.content,
                    detect_faction_from_message(message.content, channel_type),
                    author_for_logs=getattr(message.author, "name", ""),
                    message_id=referenced.id,  # map new entries to original message
                    anchor_date_str=anchor_date_str_new,
                )
                await bot.process_commands(message)
                return

    # Normal non-reply message handling
    await process_and_publish(
        message.content,
        detect_faction_from_message(message.content, channel_type),
        author_for_logs=getattr(message.author, "name", ""),
        message_id=message.id,
        anchor_date_str=message.created_at.strftime("%Y-%m-%d") if message.created_at else None,
    )

    await bot.process_commands(message)


@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message) -> None:
    if after.author == bot.user:
        return

    # Check if message is from any of our buff channels
    channel_type: Optional[str] = None
    for _, (channel_id, ctype) in BUFF_CHANNEL.items():
        if after.channel.id == channel_id:
            channel_type = ctype
            break

    if not channel_type:
        return

    # For edits, try to remove previously published entries using cache; fallback to parsing anchored to original date
    old_entries = MESSAGE_ID_TO_BUFFS.get(before.id)
    anchor_date_str = before.created_at.strftime("%Y-%m-%d") if before.created_at else None

    await process_and_publish(
        after.content,
        detect_faction_from_message(after.content, channel_type),
        is_edit=True,
        original_message=before.content,
        author_for_logs=getattr(after.author, "name", ""),
        message_id=after.id,
        anchor_date_str=after.created_at.strftime("%Y-%m-%d") if after.created_at else None,
        old_entries_to_remove=old_entries,
    )


@bot.event
async def on_message_delete(message: discord.Message) -> None:
    # Remove associated buffs if a source message is deleted
    if message.author == bot.user:
        return
    channel_type: Optional[str] = None
    for _, (channel_id, ctype) in BUFF_CHANNEL.items():
        if message.channel.id == channel_id:
            channel_type = ctype
            break
    if not channel_type:
        return

    entries = MESSAGE_ID_TO_BUFFS.pop(message.id, None)
    if not entries:
        return

    faction = detect_faction_from_message(message.content, channel_type)
    if faction:
        await remove_buffs_from_github(faction, entries)


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