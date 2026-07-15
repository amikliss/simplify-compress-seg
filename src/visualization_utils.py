import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML
import torch


def visualizeSimplification(simpDict:dict, 
                            interval=1000, name="simplification_process", save=True) -> HTML:
    """
    Generate a visualization with animation to summarize the simplification and repairing process.
    Parameters
    --------------
    input: dict
        Dictonary with the output of `SegmentRepairPipeline.do_simplification_procedure` with the keys: 
        "org_input", 
        "ref_seg", 
        "all_simp_imgs", 
        "all_simp_seg", 
        "all_rep_imgs", 
        "all_seg_rep", 
        "all_alphas"
    name: str
        File name when saving
    save: Bool
        Decide if saving or not.
    """

    # check keys
    required_keys = {"org_input", 
                     "ref_seg",
                     "all_simp_imgs",
                     "all_simp_seg",
                     "all_rep_imgs",
                     "all_seg_rep",
                     "all_alphas"}
    for key in required_keys:
        assert key in simpDict, f"Key {key} is missing"

    # First set up the figure with subplots
    # Increased width for better layout
    matplotlib.rcParams['animation.embed_limit'] = 2**128

    fig, axes = plt.subplots(3, 3, figsize=(9, 12))
    fig.suptitle("Iteration 0")

    #prepare inputs for plotting
    input = torch.moveaxis(simpDict["org_input"], 0, 2)
    reference_seg = simpDict["ref_seg"].argmax(0)

    simp_imgs = np.asarray([torch.moveaxis(img.squeeze(), 0, -1)
                           for img in simpDict["all_simp_imgs"]])
    simp_segs = simpDict["all_simp_seg"]
    
    rep_imgs = np.asarray([torch.moveaxis(img.squeeze(), 0, -1)
                            for img in simpDict["all_rep_imgs"]])
    rep_segs = simpDict["all_seg_rep"]
    
    #first frame
    org_im = axes[0, 0].imshow(input, interpolation='none')
    axes[0, 0].axis('off')
    axes[0, 0].set_title("Input Img")

    ref_seg = axes[1, 0].imshow(reference_seg)
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Reference Seg")

    simp_im = axes[0, 1].imshow(simp_imgs[0], interpolation='none')
    axes[0, 1].set_title("Simplification")
    axes[0, 1].axis('off')

    simp_seg = axes[1, 1].imshow(simp_segs[0].argmax(0), interpolation='none')
    axes[1, 1].set_title("Simplification Segmentation")
    axes[1, 1].axis('off')

    rep_im = axes[0, 2].imshow(rep_imgs[0], interpolation='none')
    axes[0, 2].set_title("Repaired")
    axes[0, 2].axis('off')

    rep_seg = axes[1, 2].imshow(rep_segs[0].argmax(0), interpolation='none')
    axes[1, 2].set_title("Repaired Segmentation")
    axes[1, 2].axis('off')

    alphas = axes[2, 0].imshow(
        simpDict["all_alphas"][0], interpolation='none', vmin= 0, vmax= 1)
    axes[2, 0].set_title("Alpha mask")
    axes[2, 0].axis('off')

    d = rep_segs[0].argmax(0) - reference_seg
    diffs = axes[2, 1].imshow(d)
    axes[2, 1].set_title("Rep_Seg - Reference")
    axes[2, 1].axis('off')

    axes[2, 2].axis('off')


    # Animation function
    def animate(i):
        fig.suptitle(f"Iteration {i}")

        simp_im.set_array(simp_imgs[i])
        axes[0, 1].set_title("Simplification")
        axes[0, 1].axis('off')

        simp_seg.set_array(simp_segs[i].argmax(0))
        axes[1, 1].set_title("Simplification Segmentation")
        axes[1, 1].axis('off')

        rep_im.set_array(rep_imgs[i])
        axes[0, 2].set_title("Repaired")
        axes[0, 2].axis('off')

        rep_seg.set_array(rep_segs[i].argmax(0))
        axes[1, 2].set_title("Repaired Segmentation")
        axes[1, 2].axis('off')

        alphas.set_array(simpDict["all_alphas"][i])
        axes[2, 0].set_title("Alpha mask")
        axes[2, 0].axis('off')

        d = rep_segs[i].argmax(0) - reference_seg
        diffs.set_array(d)
        axes[2, 1].set_title("Rep_Seg - Reference")
        axes[2, 1].axis('off')

        axes[2, 2].axis('off')

        return [simp_im, simp_seg, rep_im, rep_seg, alphas, diffs]

    anim = matplotlib.animation.FuncAnimation(fig, animate,
                                              frames=min(len(rep_imgs), len(simp_imgs)), interval=interval, blit=True)

    if save:
        # To save the animation using Pillow as a gif
        anim.save(f'{name}.gif', writer='pillow')

        FFwriter = matplotlib.animation.FFMpegWriter(fps=10)
        anim.save(f'{name}.mp4', writer=FFwriter)

    return HTML(anim.to_jshtml())


def visualize_gradients(grad_history, name="gradient_history", save=False):

    # First set up the figure with subplots
    # Increased width for better layout
    matplotlib.rcParams['animation.embed_limit'] = 2**128

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.suptitle("Iteration 0")

    grad = axes[0].imshow(grad_history[0])
    axes[0].set_title("Iteration 0")
    axes[0].axis('off')

    # mean
    grad_mean = torch.stack(grad_history).mean(dim=0)
    axes[1].imshow(grad_mean, cmap='viridis')
    axes[1].set_title('Mean gradient')

    # std
    axes[2].imshow(torch.stack(grad_history).std(dim=0), cmap='hot')
    axes[2].set_title('Gradient variance')

    def animate(i):
        fig.suptitle(f"Iteration {i}")

        grad.set_array(grad_history[i])
        axes[0].set_title(f"Iteration {i}")
        axes[0].axis('off')

        return [grad]

    anim = matplotlib.animation.FuncAnimation(fig, animate,
                                              frames=len(grad_history), interval=500, blit=True)

    if save:
        # To save the animation using Pillow as a gif
        anim.save(f'{name}.gif', writer='pillow')

        FFwriter = matplotlib.animation.FFMpegWriter(fps=10)
        anim.save(f'{name}.mp4', writer=FFwriter)

    return HTML(anim.to_jshtml())
    


