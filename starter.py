import discord
from discord.ext import commands
import logging

async def add_guild(id, data):
    data.update({str(id): []})

def squeaker():
    # Bot startup stuff
    
    logger = logging.getLogger('discord')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(handler)
    # set bot command prefix
    return commands.Bot(command_prefix='!', intents=discord.Intents.all())
