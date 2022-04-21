"""Handle the loading and initialization of game sessions."""
from __future__ import annotations

from typing import Optional
import copy
import lzma
import pickle
import traceback

from PIL import Image  # type: ignore
import tcod

from engine import Engine
from game_map import GameWorld
import color
import entity_factories
import input_handlers
import components.ai as ai
import components.config as cfg
import random


# Load the background image.  Pillow returns an object convertable into a NumPy array.
background_image = Image.open("data/menu_background.png")

def new_game(r) -> Engine:

    """Return a brand new game session as an Engine instance."""
    map_width = cfg.w #80
    map_height = cfg.h #43

    room_max_size = 20 #10
    room_min_size = 10 #6
    max_rooms = r # 30

    try:
        player = entity_factories.player # for dqn agent
    except Exception as e:
        player = copy.deepcopy(entity_factories.player)

    engine = Engine(player=player)

    engine.game_world = GameWorld(
        engine=engine,
        max_rooms=max_rooms,
        room_min_size=room_min_size,
        room_max_size=room_max_size,
        map_width=map_width,
        map_height=map_height,
    )

    engine.game_world.generate_floor()
    engine.update_fov()

    engine.message_log.add_message("Hello and welcome, adventurer, to yet another dungeon!", color.welcome_text)

    dagger = copy.deepcopy(entity_factories.dagger)
    leather_armor = copy.deepcopy(entity_factories.leather_armor)

    dagger.parent = player.inventory
    leather_armor.parent = player.inventory

    player.inventory.items.append(dagger)
    player.equipment.toggle_equip(dagger, add_message=False)

    player.inventory.items.append(leather_armor)
    player.equipment.toggle_equip(leather_armor, add_message=False)


    # auto movements
    #n = 0
    '''
    prev_score = 0

    game = True
    while game:

        if engine.player.is_alive:

            engine.handle_player_turns()

            engine.handle_enemy_turns()

            engine.update_fov()
            engine.game_map.explored |= engine.game_map.visible


            for i in engine.game_map.entities:
                # run 100 rounds on each map
                if i.name == cfg.prey and i.ai.is_new_round() and i.ai.get_rounds()%1000==0:

                    #print('==================',i.ai.get_rounds(),'======================')
                    # win rate every 100 rounds
                    print('win rate', (i.ai.get_score()-prev_score)/1000)
                    prev_score = i.ai.get_score()

                    #print('thief score:',i.ai.get_score())
                    if i.ai.get_rounds()%10000==0:
                        print('------------------------------END--------------------------')
                        #engine.player.ai.save_agent()
                        game = False
                        break




            if engine.player.level.requires_level_up:
                level_up = input_handlers.LevelUpEventHandler(engine)
        else:
            engine.update_fov()

            break

    ### aie
    '''
    return engine


def load_game(filename: str) -> Engine:
    """Load an Engine instance from a file."""

    with open(filename, "rb") as f:
        engine = pickle.loads(lzma.decompress(f.read()))
    assert isinstance(engine, Engine)
    return engine


class MainMenu(input_handlers.BaseEventHandler):
    """Handle the main menu rendering and input."""

    def __init__(self):
        self.agent_id = 0
        self.agent_num = 0

    def on_render(self, console: tcod.Console) -> None:
        """Render the main menu on a background image."""
        console.draw_semigraphics(background_image, 0, 0)

        console.print(
            console.width // 2,
            console.height // 2 - 4,
            "TOMBS OF THE ANCIENT KINGS",
            fg=color.menu_title,
            alignment=tcod.CENTER,
        )
        console.print(
            console.width // 2,
            console.height - 2,
            "By Ziyang Yu",
            fg=color.menu_title,
            alignment=tcod.CENTER,
        )

        menu_width = 24
        for i, text in enumerate(["[N] Play a new game", "[C] Continue last game", "[Q] Quit"]):
            console.print(
                console.width // 2,
                console.height // 2 - 2 + i,
                text.ljust(menu_width),
                fg=color.menu_text,
                bg=color.black,
                alignment=tcod.CENTER,
                bg_blend=tcod.BKGND_ALPHA(64),
            )

    def ev_keydown(self, event: tcod.event.KeyDown) -> Optional[input_handlers.BaseEventHandler]:
        #global agent_no
        if event.sym in (tcod.event.K_q, tcod.event.K_ESCAPE):
            raise SystemExit()
        elif event.sym == tcod.event.K_c:
            try:
                return input_handlers.MainGameEventHandler(load_game("savegame.sav"))
            except FileNotFoundError:
                return input_handlers.PopupMessage(self, "No saved game to load.")
            except Exception as exc:
                traceback.print_exc()  # Print to stderr.
                return input_handlers.PopupMessage(self, f"Failed to load save:\n{exc}")
        elif event.sym == tcod.event.K_n:

            print('=================================START===================================')
            while True:
                try:
                    r = int(input('Please enter maximum number of rooms between 1-5: '))
                except ValueError:
                    print("Sorry, the agent ID must be a number")
                    continue
                if r < 1  or r > 5:
                    print('Please select a number between 1 and 5.')
                    continue
                break
            while True:
                try:
                    print('Difficulty Level: 0\nDifficulty Level: 1\nDifficulty Level: 2')
                    print("=============================================================")
                    print('Difficulty Level 0: 1 agent of your choice per room\nDifficulty Level 1: 3 agents of your choice per room\nDifficulty Level 2: random number of random agents per room')
                    diff_level = int(input('Please choose a Difficulty Level: '))
                except ValueError:
                    print("Sorry, the agent ID must be a number")
                    continue
                else:
                    break
            if diff_level != 2:
                while True:
                    try:
                        print('Random Agent: 0\nDirectional Agent: 1\nQ-Agent: 2\nDQN Agent: 3\nMock Agent: 4')
                        MainMenu.agent_id = int(input('Please choose an agent: '))
                    except ValueError:
                        print("Sorry, the agent ID must be a number")
                        continue
                    else:
                        break
            else:
                MainMenu.agent_id = 0
            if diff_level > 2 or diff_level < 0:
                diff_level = 0 # default
            if diff_level == 0:
                MainMenu.agent_num = 1
            elif diff_level == 1:
                MainMenu.agent_num = 2
            elif diff_level == 2:
                MainMenu.agent_num = random.randint(3,5)

            if MainMenu.agent_id > 4 or MainMenu.agent_id < 0:
                MainMenu.agent_id = 0 # default
            print("=============================================================")

            return input_handlers.MainGameEventHandler(new_game(r))

    def get_agent(self):
        return MainMenu.agent_id, MainMenu.agent_num

        return None
