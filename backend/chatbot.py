import aiml
import os

kernel = aiml.Kernel()

BASE = os.path.dirname(__file__)
kernel.learn(os.path.join(BASE, "basic_chat.aiml"))

def get_bot_response(label):
    return kernel.respond(label)
