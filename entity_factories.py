from components import consumable, equippable
from components.ai import HostileEnemy, CompetitiveEnemy, ThiefEnemy, ThiefEnemy2, ThiefEnemy3, DirectionalEnemy, MockEnemy
from components.equipment import Equipment
from components.fighter import Fighter
from components.inventory import Inventory
from components.level import Level
from entity import Actor, Item

import sys
sys.tracebacklimit=0

player = Actor(
    char="@",
    color=(255, 255, 255),
    name="Player",
    ai_cls=HostileEnemy,
    #ai_cls=ThiefEnemy2, #training
    #ai_cls = ThiefEnemy, # comparing
    equipment=Equipment(),
    fighter=Fighter(hp=20, base_defense=1, base_power=2),
    #fighter=Fighter(hp=1, base_defense=0, base_power=0), # for training
    inventory=Inventory(capacity=0),
    level=Level(level_up_base=200),
)

miner1 = Actor(
    char="$",
    color=(25, 127, 60),
    name="Miner",
    #ai_cls=CompetitiveEnemy, #training
    ai_cls=DirectionalEnemy,
    #ai_cls=ThiefEnemy,
    equipment=Equipment(),
    fighter=Fighter(hp=20, base_defense=1, base_power=0),
    inventory=Inventory(capacity=0),
    level=Level(xp_given=100),
)

# random
miner0 = Actor(
    char="$",
    color=(25, 127, 60),
    name="Miner",
    #ai_cls=CompetitiveEnemy, #training
    ai_cls=ThiefEnemy,
    #ai_cls=ThiefEnemy,
    equipment=Equipment(),
    fighter=Fighter(hp=20, base_defense=1, base_power=0),
    inventory=Inventory(capacity=0),
    level=Level(xp_given=100),
)

# q agent
miner2 = Actor(
    char="$",
    color=(25, 127, 60),
    name="Miner",
    #ai_cls=CompetitiveEnemy, #training
    ai_cls=ThiefEnemy2,
    #ai_cls=ThiefEnemy,
    equipment=Equipment(),
    fighter=Fighter(hp=20, base_defense=1, base_power=0),
    inventory=Inventory(capacity=0),
    level=Level(xp_given=100),
)

# dqn agent
miner3 = Actor(
    char="$",
    color=(25, 127, 60),
    name="Miner",
    #ai_cls=CompetitiveEnemy, #training
    ai_cls=ThiefEnemy3,
    #ai_cls=ThiefEnemy,
    equipment=Equipment(),
    fighter=Fighter(hp=20, base_defense=1, base_power=0),
    #fighter=Fighter(hp=10, base_defense=10, base_power=10),
    inventory=Inventory(capacity=0),
    level=Level(xp_given=100),
)

# mock agent
miner4 = Actor(
    char="$",
    color=(25, 127, 60),
    name="Miner",
    #ai_cls=CompetitiveEnemy, #training
    ai_cls=MockEnemy,
    #ai_cls=ThiefEnemy,
    equipment=Equipment(),
    fighter=Fighter(hp=20, base_defense=1, base_power=0),
    #fighter=Fighter(hp=10, base_defense=10, base_power=10),
    inventory=Inventory(capacity=0),
    level=Level(xp_given=100),
)

orc = Actor(
    char="o",
    color=(63, 127, 63),
    name="Orc",
    ai_cls=HostileEnemy,
    equipment=Equipment(),
    fighter=Fighter(hp=10, base_defense=0, base_power=3),
    inventory=Inventory(capacity=0),
    level=Level(xp_given=35),
)

troll = Actor(
    char="T",
    color=(0, 127, 0),
    name="Troll",
    ai_cls=HostileEnemy,
    equipment=Equipment(),
    fighter=Fighter(hp=16, base_defense=1, base_power=4),
    inventory=Inventory(capacity=0),
    level=Level(xp_given=100),
)

confusion_scroll = Item(
    char="~",
    color=(207, 63, 255),
    name="Confusion Scroll",
    consumable=consumable.ConfusionConsumable(number_of_turns=10),
)
fireball_scroll = Item(
    char="~",
    color=(255, 0, 0),
    name="Fireball Scroll",
    consumable=consumable.FireballDamageConsumable(damage=12, radius=3),
)
health_potion = Item(
    char="!",
    color=(127, 0, 255),
    name="Health Potion",
    consumable=consumable.HealingConsumable(amount=10),
)
lightning_scroll = Item(
    char="~",
    color=(255, 255, 0),
    name="Lightning Scroll",
    consumable=consumable.LightningDamageConsumable(damage=20, maximum_range=5),
)

dagger = Item(char="/", color=(0, 191, 255), name="Dagger", equippable=equippable.Dagger())

sword = Item(char="/", color=(0, 191, 255), name="Sword", equippable=equippable.Sword())

leather_armor = Item(
    char="[",
    color=(139, 69, 19),
    name="Leather Armor",
    equippable=equippable.LeatherArmor(),
)

chain_mail = Item(char="[", color=(139, 69, 19), name="Chain Mail", equippable=equippable.ChainMail())
