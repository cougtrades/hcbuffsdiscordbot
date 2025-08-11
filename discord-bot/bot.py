import os
import re
import discord
from discord.ext import commands
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from openai import OpenAI
import json
import time
import random
import sys
import base64
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv('OPENAIKEY'))

# Set up bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Configuration for the buff channels
BUFF_CHANNEL = {
    'horde': (1306953557378859119, 'horde'),  # Horde channel
    'alliance': (1336750424048271381, 'alliance'),  # Alliance channel
    'both': (1366901374826315837, 'both')  # Zandalar channel
}

# Note: Currently all channels are using the same ID (1366901374826315837) which is the 🐲world-buffs-zandalar channel
# This means the bot will only monitor one channel. To monitor separate channels, we need the correct channel IDs for:
# - Horde channel
# - Alliance channel
# - Both channel

def parse_buff_with_ai(message_content):
    """Use OpenAI to parse buff messages and format them for the buff system"""
    try:
        print("\nParsing message with OpenAI...")
       
        # Get today's date for default date handling
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
       
        # Create a prompt that explains what we want
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
   - If no date is specified in the message, use today's date ({today_str})
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
   - When no date is specified, use today's date ({today_str})

Example Conversions:
- "1800 ST" → Use today's date + 6 hours
- "8:15 ST" → Use today's date + 6 hours
- "8:15 ST 6/10" → "2025-06-11T02:15:00Z" (8:15 PM MT + 6 hours = 02:15 next day UTC)
- "7:45 ST 6/11" → "2025-06-12T01:45:00Z" (7:45 PM MT + 6 hours = 01:45 next day UTC)
- "7:47 ST 6/11 (have nef backup)" → "2025-06-12T01:47:00Z" (7:47 PM MT + 6 hours = 01:47 next day UTC)
- "6:55 PM ST" → Use today's date + 6 hours
- "6:58 PM ST" → Use today's date + 6 hours

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

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": f"""You are a WoW buff message parser that handles various human-written formats.
                Your task is to extract buff information and convert times to UTC/Z format.
                Key rules:
                1. Server Time (ST) is Mountain Time
                2. Convert all times to UTC/Z by adding 6 hours
                3. Use next day's date for times 18:00-23:59 MT
                4. Use same day's date for times 00:00-17:59 MT
                5. If no date is specified, use today's date ({today_str})
                6. Always use server name Doomhowl
                7. Handle DST: After March 9, 2025, use 6-hour offset; before that, use 7-hour offset
                8. Standardize buff names to: Zandalar, Nefarian, Rend, or Onyxia
                9. Be flexible with time formats - handle both 12/24 hour, various separators, and AM/PM indicators
                10. Both Horde and Alliance use the same time conversion rules
                11. Example: 6:55 PM ST becomes 00:55 UTC next day (not 01:55)"""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for more consistent results
        )

        # Extract the JSON response and clean it
        json_str = response.choices[0].message.content.strip()
        print(f"AI Response: {json_str}")
       
        # Remove markdown code block markers if present
        json_str = json_str.replace('```json', '').replace('```', '').strip()
       
        # Parse the JSON
        buff_data_list = json.loads(json_str)
       
        return buff_data_list

    except Exception as e:
        print(f"❌ Error parsing with AI: {str(e)}")
        return None

@bot.event
async def on_ready():
    print(f'Bot is ready! Logged in as {bot.user.name}')
   
    # Monitor both channels
    for faction, (channel_id, faction_name) in BUFF_CHANNEL.items():
        channel = bot.get_channel(channel_id)
        if channel:
            print(f'Monitoring {faction} channel: {channel.name}')
           
            async for message in channel.history(limit=1):
                if message.author != bot.user:
                    print(f'\nProcessing last message from {message.author.name}:')
                    print(f'Content: {message.content}')
                   
                    # Detect faction based on channel type
                    detected_faction = detect_faction_from_message(message.content, faction)
                    if not detected_faction:
                        print("Could not detect faction from message content")
                        continue
                   
                    buff_data_list = parse_buff_with_ai(message.content)
                   
                    if buff_data_list:
                        print(f'\nFound {len(buff_data_list)} buffs:')
                        for buff_data in buff_data_list:
                            print(f"- {buff_data['buff']} by {buff_data['guild']} at {buff_data['datetime']}")
                            buff_data['faction'] = detected_faction
                            await publish_buff_to_github(buff_data)
        else:
            print(f'❌ Channel not found (ID: {channel_id})')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if message is from any of our buff channels
    channel_faction = None
    for faction, (channel_id, faction_name) in BUFF_CHANNEL.items():
        if message.channel.id == channel_id:
            detected_faction = detect_faction_from_message(message.content, faction)
            if not detected_faction:
                print("Could not detect faction from message content")
                return
            channel_faction = detected_faction
            break

    if not channel_faction:
        return

    print(f'\nProcessing message from {message.author.name}:')
    print(f'Content: {message.content}')
   
    buff_data_list = parse_buff_with_ai(message.content)
   
    if buff_data_list:
        print(f'\nFound {len(buff_data_list)} buffs:')
        for buff_data in buff_data_list:
            print(f"- {buff_data['buff']} by {buff_data['guild']} at {buff_data['datetime']}")
            buff_data['faction'] = channel_faction
            await publish_buff_to_github(buff_data)
   
    await bot.process_commands(message)

@bot.event
async def on_message_edit(before, after):
    """Handle edited messages the same way as new messages"""
    if after.author == bot.user:
        return

    # Check if message is from any of our buff channels
    channel_faction = None
    for faction, (channel_id, faction_name) in BUFF_CHANNEL.items():
        if after.channel.id == channel_id:
            detected_faction = detect_faction_from_message(after.content, faction)
            if not detected_faction:
                print("Could not detect faction from message content")
                return
            channel_faction = detected_faction
            break

    if not channel_faction:
        return

    print(f"\nEdited message from {after.author.name}:")
    print(f"Before: {before.content}")
    print(f"After: {after.content}")
   
    buff_data_list = parse_buff_with_ai(after.content)
   
    if buff_data_list:
        print(f"\nFound {len(buff_data_list)} buffs:")
        for buff_data in buff_data_list:
            print(f"- {buff_data['buff']} by {buff_data['guild']} at {buff_data['datetime']}")
            buff_data['faction'] = channel_faction
            await publish_buff_to_github(buff_data, is_edit=True, original_message=before.content)
    else:
        print("No buffs found in edited message or error occurred")
        if any(keyword in after.content.lower() for keyword in ['zg', 'ony', 'rend', 'nef']):
            await after.channel.send("Could not parse buffs from edited message. Please use format: <Guild> MM.DD.YY <emoji> HH:MM AM/PM")

@bot.command(name='test_parse')
async def test_parse(ctx, *, message_content):
    """Test the message parser with custom content"""
    buff_data_list = parse_buff_with_ai(message_content)
    if buff_data_list:
        response = f"Found {len(buff_data_list)} buffs:\n"
        for i, buff_data in enumerate(buff_data_list, 1):
            response += f"{i}. **{buff_data['buff']}** by *{buff_data['guild']}* at `{buff_data['datetime']}`\n"
    else:
        response = "No buffs found in the message."
   
    await ctx.send(response)

async def publish_buff_to_github(buff_data, is_edit=False, original_message=None):
    """Publish a buff to GitHub using the same logic as add-buff.js"""
    try:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("❌ Error: GitHub token not configured")
            return False

        faction = buff_data['faction']
        # Ensure we use the correct filename format
        if faction not in ['horde', 'alliance']:
            print(f"❌ Invalid faction: {faction}")
            return False
            
        path = f"{faction}_buffs.json"
       
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
       
        # Get current file content
        url = f'https://api.github.com/repos/cougtrades/wowbuffs/contents/{path}'
        file_response = requests.get(url, headers=headers)
       
        if not file_response.ok:
            print(f"❌ Failed to fetch file: {file_response.text}")
            return False
           
        file_data = file_response.json()
        current_content = json.loads(base64.b64decode(file_data['content']).decode('utf8'))
       
        # Create new buff entry
        new_buff = {
            'datetime': buff_data['datetime'],
            'guild': buff_data['guild'],
            'buff': buff_data['buff'],
            'notes': buff_data.get('notes', ''),
            'server': 'Doomhowl'
        }

        # Check for duplicates
        if is_duplicate_buff(new_buff, current_content):
            print(f"⚠️ Skipping duplicate buff: {new_buff['guild']} {new_buff['buff']} at {new_buff['datetime']}")
            return True

        # If this is an edit, parse the original message and remove the old entry
        if is_edit and original_message:
            original_buff_data_list = parse_buff_with_ai(original_message)
            if original_buff_data_list:
                original_buff = original_buff_data_list[0]
                # Remove the old entry that matches the original message
                current_content = [
                    buff for buff in current_content 
                    if not (buff['datetime'] == original_buff['datetime'] and 
                           buff['guild'].lower() == original_buff['guild'].lower() and 
                           buff['buff'] == original_buff['buff'])
                ]

        # Add new buff and sort
        current_content.append(new_buff)
        current_content.sort(key=lambda x: x['datetime'])
       
        # Update file on GitHub
        update_response = requests.put(
            f'https://api.github.com/repos/cougtrades/wowbuffs/contents/{path}',
            headers=headers,
            json={
                'message': f'Update {path} with new buff from Discord: {new_buff["guild"]} {new_buff["buff"]}',
                'content': base64.b64encode(json.dumps(current_content, indent=4).encode()).decode(),
                'sha': file_data['sha']
            }
        )
       
        if not update_response.ok:
            print(f"❌ Failed to update file: {update_response.text}")
            return False
           
        print(f"✅ Published buff: {new_buff['guild']} {new_buff['buff']}")
        return True
       
    except Exception as e:
        print(f"❌ Error publishing to GitHub: {str(e)}")
        return False

def detect_faction_from_message(message_content, channel_type=None):
    """Detect faction from message content based on channel type and emojis"""
    # If it's the "both" channel, we need to detect from emojis
    if channel_type == 'both':
        if ':horde:' in message_content.lower():
            return 'horde'
        elif ':alliance:' in message_content.lower():
            return 'alliance'
        return None
    
    # For horde and alliance channels, use the channel type
    return channel_type

def is_duplicate_buff(buff_data, current_content):
    """Check if a buff is a duplicate based on datetime, guild, and buff type"""
    for existing_buff in current_content:
        if (existing_buff['datetime'] == buff_data['datetime'] and
            existing_buff['guild'].lower() == buff_data['guild'].lower() and
            existing_buff['buff'] == buff_data['buff']):
            return True
    return False

def main():
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        print('Error: DISCORD_TOKEN not found in .env file')
        return
   
    if not os.getenv('OPENAIKEY'):
        print('Error: OPENAIKEY not found in .env file')
        return
   
    if not os.getenv('GITHUB_TOKEN'):
        print('Error: GITHUB_TOKEN not found in .env file')
        return
   
    try:
        bot.run(token)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        print(f"Error running bot: {e}")

if __name__ == '__main__':
    main()