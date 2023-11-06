import matplotlib.pyplot as plt
from cppn_torch.util import image_grid


def show_move_map_images(move):
    pop = move.map.get_population()
    imgs = [g(move.inputs, channel_first=False).detach().cpu() for g in pop]
    image_grid(
        imgs,
        cols=10,
        titles=[f"{move.map.cell_names[i]}\n{g.id}:{g.fitness.item()}" for i, g in enumerate(pop)],
        show=True,
        fig_size=(30,30),
        suptitle=f"Generation {move.gen}",
        title_font_size=8,
    )
    
    
def get_dynamic_mut_rate(rate, run_progress, end_mod):
    return rate - (rate - end_mod * rate) * run_progress