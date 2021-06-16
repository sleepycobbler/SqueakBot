import discord
import logging
from discord.ext import commands
import random
import json
import nltk

# Bot startup stuff
bot_token = 'bottokengoeshere'
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)
# set bot command prefix
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

victim = 227949546896883712

# Sets the game status / states when the bot is online
@bot.event
async def on_ready():
    print('We have logged in as {0.user}'.format(bot))
    game = discord.Game("your ass")
    await bot.change_presence(activity=game, status=discord.Status.online)

# Grab a message from the list and slam it in chat
@bot.command(name="ben")
async def ben(ctx):
    async with ctx.typing():
        with open("ben.json", "r+") as f:
            data = json.load(f)
            # if the guild hasn't used the bot before, we gotta add it to the list
            if str(ctx.guild.id) not in data.keys():
                data.update({str(ctx.guild.id): []})
                channels = ctx.guild.text_channels
                members = ctx.guild.members
                for channel in channels:
                    # this single line of code take 99% longer than any other line here
                    messages = await channel.history(limit=None).flatten()
                    for message in messages:
                        # Check if it is the victim, and if it isn't an image / file / something I cant parse
                        if message.author.id == victim and len(message.content) > 0:
                            # Make sure its not a bot command (THIS WOULD BE REALLY BAD TO REMOVE)
                            if message.content[0] not in ['-', '!', '>'] and not message.content.startswith("toddbot"):
                                data[str(ctx.guild.id)].append(message.content)
                for range in data[str(ctx.guild.id)]:
                    message = random.choice(data[str(ctx.guild.id)])
                    if message != "":
                        await ctx.channel.send(message)
                        break
                # Delete Everything
                f.truncate(0)
                # Set cursor to correct place
                f.seek(0)
                # Delete Everything again?
                f.flush()
                # get rid of all duplicate messages
                data[str(ctx.guild.id)] = list(dict.fromkeys(data[str(ctx.guild.id)]))
                # Profit (stored in thing)
                f.write(json.dumps(data))
            else:
                for range in data[str(ctx.guild.id)]:
                    message = random.choice(data[str(ctx.guild.id)])
                    if message != "":
                        await ctx.channel.send(message)
                        break

# get all messages from the server and filter them
@bot.command(name="updateben")
async def updatebot(ctx):
    async with ctx.typing():
        with open("ben.json", "r+") as f:
            data = json.load(f)
            if str(ctx.guild.id) not in data.keys():
                data.update({str(ctx.guild.id): []})
                channels = ctx.guild.text_channels
                for channel in channels:
                    messages = await channel.history(limit=None).flatten()
                    for message in messages:
                        if message.author.id == victim and len(message.content) > 0:
                            if message.content[0] not in ['-', '!', '>'] and not message.content.startswith("toddbot"):
                                data[str(ctx.guild.id)].append(message.content)
                f.truncate(0)
                f.seek(0)
                f.flush()
                data[str(ctx.guild.id)] = list(dict.fromkeys(data[str(ctx.guild.id)]))
                f.write(json.dumps(data))
                await ctx.channel.send("BEN DATA HAS BEEN PROCESSED AND STORED")
            else:
                channels = ctx.guild.text_channels
                for channel in channels:
                    messages = await channel.history(limit=None).flatten()
                    for message in messages:
                        if message.author.id == victim and len(message.content) > 0:
                            if message.content[0] not in ['-', '!']:
                                data[str(ctx.guild.id)].append(message.content)
                f.truncate(0)
                f.seek(0)
                f.flush()
                data[str(ctx.guild.id)] = list(dict.fromkeys(data[str(ctx.guild.id)]))
                f.write(json.dumps(data))
                await ctx.channel.send("BEN DATA HAS BEEN PROCESSED AND STORED")

# get a ben message that is not a link
@bot.command(name="ben_nolinks")
async def bot_no_links(ctx):
    async with ctx.typing():
        with open("ben.json", "r+") as f:
            data = json.load(f)
            if str(ctx.guild.id) not in data.keys():
                data.update({str(ctx.guild.id): []})
                channels = ctx.guild.text_channels
                for channel in channels:
                    messages = await channel.history(limit=None).flatten()
                    for message in messages:
                        if message.author.id == victim and len(message.content) > 0:
                            if message.content[0] not in ['-', '!', '>'] and not message.content.startswith("toddbot"):
                                data[str(ctx.guild.id)].append(message.content)
                for range in data[str(ctx.guild.id)]:
                    message = random.choice(data[str(ctx.guild.id)])
                    if message != "" and not message.startswith('http'):
                        await ctx.channel.send(message)
                        break
                f.truncate(0)
                f.seek(0)
                f.flush()
                data[str(ctx.guild.id)] = list(dict.fromkeys(data[str(ctx.guild.id)]))
                f.write(json.dumps(data))
            else:
                for range in data[str(ctx.guild.id)]:
                    message = random.choice(data[str(ctx.guild.id)])
                    if message != "" and not message.startswith('http'):
                        await ctx.channel.send(message)
                        break

# Get a ben message that is a link
@bot.command(name="ben_link")
async def bot_links(ctx):
    async with ctx.typing():
        with open("ben.json", "r+") as f:
            data = json.load(f)
            if str(ctx.guild.id) not in data.keys():
                data.update({str(ctx.guild.id): []})
                channels = ctx.guild.text_channels
                for channel in channels:
                    messages = await channel.history(limit=None).flatten()
                    for message in messages:
                        if message.author.id == victim and len(message.content) > 0:
                            if message.content[0] not in ['-', '!', '>'] and not message.content.startswith("toddbot"):
                                data[str(ctx.guild.id)].append(message.content)
                for range in data[str(ctx.guild.id)]:
                    message = random.choice(data[str(ctx.guild.id)])
                    if message != "" and message.startswith('http'):
                        await ctx.channel.send(message)
                        break
                f.truncate(0)
                f.seek(0)
                f.flush()
                data[str(ctx.guild.id)] = list(dict.fromkeys(data[str(ctx.guild.id)]))
                f.write(json.dumps(data))
            else:
                for range in data[str(ctx.guild.id)]:
                    message = random.choice(data[str(ctx.guild.id)])
                    if message != "" and message.startswith('http'):
                        await ctx.channel.send(message)
                        break

########################################################################################################################
# "What is this?" you may ask. This, my friend, is the amalgamation of about 5 hours of extremely ineffective research
# boiled down to "Eh this works I guess". The Natural Language Toolkit is able to do much more, and maybe sometime
# I'll actually figure out how to do it. But for now, all it does is it takes every word, and keys it to every possible
# word that comes before and after. It's not actually "ai", but eh.
########################################################################################################################
@bot.command(name="ben_ai")
async def bot_talk(ctx):
    with open("ben.json", "r+") as f:
        data = json.load(f)
    words = []
    for guild in data.keys():
        for message in data[guild]:
            if message != "" and not message.startswith('http') and not message.startswith('<'):
                message_alt = message
                for word in nltk.word_tokenize(message_alt):
                    words.append(word)
    ben_text = nltk.text.Text(words)
    await ctx.channel.send(ben_text.generate(length=random.randrange(5, 100), random_seed=random.randint(0, 9999999999999999999999)))

bot.run(bot_token)
