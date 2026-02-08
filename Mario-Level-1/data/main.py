__author__ = 'justinarmstrong'

from . import setup,tools
from .states import main_menu,load_screen,level1
from . import constants as c


# Global face swap data shared across game states
face_swap_data = {
    'styled_face': None,  # BGRA numpy array of processed face
    'style_name': None,   # 'pixel', 'original', or 'cartoon'
    'enabled': False,
}


def main():
    """Add states to control here."""
    # Show face swap UI before starting the game
    from .face_swap.ui import FaceSwapUI
    screen = setup.SCREEN
    ui = FaceSwapUI(screen)
    styled_face, style_name = ui.run()

    if styled_face is not None:
        face_swap_data['styled_face'] = styled_face
        face_swap_data['style_name'] = style_name
        face_swap_data['enabled'] = True

    run_it = tools.Control(setup.ORIGINAL_CAPTION)
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.TIME_OUT: load_screen.TimeOut(),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.LEVEL1: level1.Level1()}

    run_it.setup_states(state_dict, c.MAIN_MENU)
    run_it.main()



