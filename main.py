import discord
from discord.ext import commands

import starter
import grabber
import ai

bot_token = 'bottokenhere'
bot = starter.squeaker()

# Sets the game status / states when the bot is online
@bot.event
async def on_ready():
    print('We have logged in as {0.user}'.format(bot))
    game = discord.Game("your ass")
    await bot.change_presence(activity=game, status=discord.Status.online)

@bot.command(name="update_squeak") # get all messages from the server and filter them
async def updatebot(ctx):
    async with ctx.typing():
        await grabber.update(ctx)

@bot.command(name="squeak") # Grab a message from the list and slam it in chat
async def bot_grab(ctx):
    async with ctx.typing():
        await grabber.grabber(ctx)

@bot.command(name="squeak_nolinks") # get a message that is not a link
async def bot_no_links(ctx):
    async with ctx.typing():
        await grabber.grabber(ctx, links=False)

@bot.command(name="squeak_link") # Get a message that is a link
async def bot_links(ctx):
    async with ctx.typing():
        await grabber.grabber(ctx, links=True)

@bot.command(name="squeak_ai")
async def bot_talk(ctx):
    async with ctx.typing():
        await ai.ai_gen_tf(ctx)

bot.run(bot_token)
