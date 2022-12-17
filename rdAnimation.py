import sys
import os
import pickle
import numpy as np
from numpy.random import rand
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib
from tqdm import tqdm

#import MultiReactionDiffusion
from MultiReactionDiffusion import MultiReactionDiffusion

def model_animation(model, size, init_voronoi, updates_per_frame):
    model.init_lattices(size, init_voronoi)
    
    # SETUP ANIMATION --------------------------------------
    fig, ax = plt.subplots()
    ax.set_facecolor('black')

    colourmap = cm.get_cmap('gist_rainbow')
    implot = ax.imshow(model.output_lattice, cmap=colourmap, interpolation='nearest', alpha=1.0*model.output_alphas + 0.0)   
    fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0,vmax=(model.n_species-1)) , cmap=colourmap), ax=ax)
        
    concentration_data = []
    mean_interspecies_variances = []
    concentration_spatial_variances = []
    kernel = np.array([[0,0,0],[0,-2,1],[0,1,0]])

    def update(frame):
        prev_concentrations = np.copy(model.species_concentrations()) # <- keep this for checking if we think it has terminated

        for i in range(updates_per_frame):
            model.update()
        model.update_visual()

        ### occasionally store data to plot at end, dont need to do this every frame
        if (frame % 5 == 0):
            concentration_data.append(model.species_concentrations())

            mean_interspecies_variances.append(np.mean(np.var(model.species_lattices, axis=0)))

            values = np.zeros(model.n_species)
            for s in range(model.n_species):
                values[s] = np.mean(np.abs(convolve(model.species_lattices[s,:,:], kernel, mode='wrap')))
            concentration_spatial_variances.append(values)
        
        ### log iteration number sometimes
        if (frame % 100 == 0 and frame != 0): 
            print("iteration " + str(frame))
            #print()
            if np.all(np.abs(model.species_concentrations() - prev_concentrations) < 0.00001):
                print("finished")

        implot.set_data(model.output_lattice)
        implot.set(alpha=1.0*model.output_alphas + 0.0)
    
    
    ani = animation.FuncAnimation(fig, update, interval=10)
    
    plt.show()
    
    # SAVE THE MODEL ---------------------------------------
    x = input("save the model? (y/n): ")
    if x == "y" or x == "Y":
        file_name = input("enter model name: ")
        print("saving the model")
        file = open(file_name, 'wb')
        pickle.dump(model, file)
        file.close()
    else:
        print("not saving")


    ### plot graph after simulation animation: --------------

    concentration_data = np.array(concentration_data)
    mean_interspecies_variances = np.array(mean_interspecies_variances)
    concentration_spatial_variances = np.array(concentration_spatial_variances)

    fig, axs = plt.subplots(ncols=3, nrows=1)
    
    cmapvals = np.linspace(0,1,model.n_species)
    for i in range(model.n_species):
        axs[0].plot(concentration_data[:,i], color=colourmap(cmapvals[i]))
        axs[2].plot(concentration_spatial_variances[:,i], color=colourmap(cmapvals[i]))
    axs[0].legend([str(i) for i in range(model.n_species)])
    axs[0].set_ylabel("concentrations")
    axs[1].plot(mean_interspecies_variances)
    axs[1].set_ylabel("mean inter-species variances")
    axs[2].legend([str(i) for i in range(model.n_species)])
    axs[2].set_ylabel("concentration spatial variances")
    plt.show()


def load_model(model_name):
    file = open(model_name, 'rb')
    model = pickle.load(file)
    file.close()
    return model

# Main entry point of the program
if __name__ == "__main__":
    
    # Read input arguments
    args = sys.argv
    size = int(args[1])
    n_species = int(args[2])
    order = int(args[3])

    #model = MultiReactionDiffusion(size, n_species, order)
    model = load_model('interesting_patterns')
    model_animation(model, size, False, 5)