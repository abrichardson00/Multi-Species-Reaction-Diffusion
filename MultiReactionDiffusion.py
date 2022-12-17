import copy
import numpy as np
from numpy.random import rand
from scipy.ndimage import convolve
from itertools import combinations

class MultiReactionDiffusion(object):
    
    def __init__(self, size, n_species, order):
        self.size = size
        self.dx = 1.0 # = dy 
        self.dt = 1.0
        self.D = 0.02

        self.order = order
        self.n_species = n_species

        #self.genes = np.zeros(sum([self.n_species**i for i in range(self.order+1)]))
        species_list = [i for i in range(self.n_species)]
        self.gene_species_lookup = [list(combinations(species_list,i+1)) for i in range(self.order)]
        self.gene_species_lookup = sum(self.gene_species_lookup, [])
        self.gene_species_lookup = [list(x) for x in self.gene_species_lookup]

        # random initialization: between -0.5 and 0.5
        self.genes = [1.0*(rand(len(self.gene_species_lookup))-0.5) for i in range(self.n_species)]
        # extra optional normalization effect: can be handy to improve chances of visually interesting interactions:
        #self.genes = [(0.5*len(self.gene_species_lookup)*gene) / np.sum(np.abs(gene)) for gene in self.genes]
        
        # alternative: initialize with zeros (then call mutate later to randomize)
        #self.genes = [np.zeros(len(self.gene_species_lookup)) for i in range(self.n_species)]

        self.init_voronoi = False
        self.init_lattices(self.size, self.init_voronoi)

        self.output_lattice = np.zeros((self.size, self.size), dtype=int)
        self.output_alphas = np.zeros((self.size, self.size), dtype=float)

        # kernel for the finite difference scheme for updating the simulation
        self.kernel = np.array([[0, 1, 0],
                                [1,-4, 1],
                                [0, 1, 0]])
        self.update_view_lattices()


    def init_lattices(self, size, init_voronoi):
        self.size = size
        # reset these incase size is changed
        self.output_lattice = np.zeros((self.size, self.size), dtype=int)
        self.output_alphas = np.zeros((self.size, self.size), dtype=float)

        if init_voronoi:
            self.species_lattices = np.zeros((self.n_species, self.size, self.size))
            n_points = [ np.array([rand()*self.size, rand()*self.size]) for i in range(self.n_species)]
            for s in range(self.n_species):
                for x in range(self.size):
                    for y in range(self.size):
                        self.species_lattices[s, x, y] = np.linalg.norm(n_points[s] - np.array([x,y]))
        else:
            self.species_lattices = np.array([(1.0/self.n_species)*rand(self.size, self.size) - (0.0/self.n_species) for i in range(self.n_species)])
        
        sum_lattice = np.sum(np.abs(self.species_lattices), axis=0)
        for s in range(self.n_species):
            self.species_lattices[s,:,:] = np.divide(self.species_lattices[s,:,:], sum_lattice)
        self.update_view_lattices()


    
    # gene_species_lookup stores a mapping from gene term to the RD species it is interacting with.
    # i.e. with order one, gene lookup is just [[1], [2], [3], [4] ...], but with order > 1 we can
    # have [[1], [1, 2], [1, 3], [2], ... etc] 

    def update(self):
        self.prev_species_lattices = np.copy(self.species_lattices)
        for s in range(self.n_species):
            self.species_lattices[s,:,:] = self.species_lattices[s,:,:] + (((self.D/(self.dx**2))*convolve(self.prev_species_lattices[s,:,:], self.kernel, mode='wrap')))*self.dt

            total = np.zeros((self.size, self.size), dtype='float64')
            for i in range(len(self.gene_species_lookup)):
                per_gene = np.ones((self.size, self.size))
                for j in range(len(self.gene_species_lookup[i])):
                    per_gene = per_gene * self.prev_species_lattices[self.gene_species_lookup[i][j],:,:]
                total = total + (self.genes[s][i]*(len(self.gene_species_lookup[i])**(len(self.gene_species_lookup[i])))*per_gene)

            # need to process 'total' in some way to keep things nicely bounded between 0 and 1.
            # multiply by logistic map is a neat approach           
            total = total*(self.prev_species_lattices[s,:,:]*(1 - self.prev_species_lattices[s,:,:]))

            # other approaches can be tried:
            #total = total*(self.n_species / (self.genes[s].shape[0]))
            #total = total*(0.25 - (0.5 - self.prev_species_lattices[s,:,:])**2)
            
            ### actually update the species lattice now!
            self.species_lattices[s,:,:] = self.species_lattices[s,:,:] + total*self.dt
            # incase an update technically makes it below 0 (i.e. numerical / rounding errors etc):
            self.species_lattices[s][self.species_lattices[s,:,:] < 0.0] = 0.0
        
        # now normalize each lattice point by the sum of all species values - i.e. concentrations of each species
        sum_lattice = np.sum(np.abs(self.species_lattices), axis=0)
        for s in range(self.n_species):
            self.species_lattices[s,:,:] = np.divide(self.species_lattices[s,:,:], sum_lattice)

    
    ### for animation
    def update_view_lattices(self):
        for i in range(self.n_species):
            indices = np.ones((self.size, self.size), dtype=bool)
            for j in range(self.n_species):
                if i != j:
                    indices = indices & (self.species_lattices[i,:,:] > self.species_lattices[j,:,:])
            if np.any(indices):
                self.output_lattice[indices] = i
                self.output_alphas[indices] = np.sqrt(self.species_lattices[i][indices])
    
    ### for animation
    def update_visual(self):
        self.update()
        self.update_view_lattices()

    ### for any updating with genetic algorithms
    def update_genes_with_copy(self, copied_index, replaced_index):
        for i in range(len(self.gene_species_lookup)):
            lookup = copy.deepcopy(self.gene_species_lookup[i])
            if replaced_index in lookup:
                lookup.remove(replaced_index)
                if not (copied_index in lookup):
                    lookup.append(copied_index)
                lookup_index = None
                for x in range(len(self.gene_species_lookup)):
                    if all([val in self.gene_species_lookup[x] for val in lookup]):
                        lookup_index = x
                # lookup_index = self.gene_species_lookup.index(lookup)
                for s in range(self.n_species):
                    self.genes[s][i] = self.genes[s][lookup_index]

    def simulate_batch(self, n_sims, n_values, iterations):
        fitness_data = np.zeros((self.n_species,n_sims,n_values))
        for n in range(n_sims):
            self.init_lattices(self.size, self.init_voronoi)
            for i in range(50):
                # some equilibration time
                prev_concentrations = np.copy(self.species_concentrations())
                self.update()
                if np.all(np.abs(self.species_concentrations() - prev_concentrations) < 0.00001):
                    break

            for i in range(n_values):
                for j in range(iterations//n_values):
                    self.update()
                fitness_data[:,n,i] = self.species_concentrations()

        fitness_means = np.mean(fitness_data, axis=2)
        return np.mean(fitness_means, axis=1)

    def get_metrics(self, n_sims, n_values, iterations):

        concentrations = np.zeros((self.n_species, n_sims, n_values))
        concentration_temporal_variances = np.zeros((self.n_species, n_sims))
        mean_interspecies_variances = np.zeros((n_sims, n_values))
        concentration_spatial_variances = np.zeros((self.n_species, n_sims, n_values))
        kernel = np.array([[0,0,0],[0,-2,1],[0,1,0]])
        for n in range(n_sims):
            self.init_lattices(self.size, self.init_voronoi)
            for i in range(100):
                # some equilibration time
                self.update()

            for i in range(n_values):
                for j in range(iterations//n_values):
                    self.update()
                concentrations[:,n,i] = self.species_concentrations()
                mean_interspecies_variances[n,i] = np.mean(np.var(self.species_lattices, axis=0))
                for s in range(self.n_species):
                    concentration_spatial_variances[s,n,i] = np.mean(np.abs(convolve(self.species_lattices[s,:,:], kernel, mode='wrap')))
            concentration_temporal_variances[:, n] = np.var(concentrations[:,n,:], axis=1)
        return np.mean(np.mean(concentrations, axis=2), axis=1), np.mean(concentration_temporal_variances, axis=1), np.mean(mean_interspecies_variances), np.mean(np.mean(concentration_spatial_variances, axis=2), axis=1)

    def mutate_species(self, s, n):
        n_genes = len(self.genes[s])
        for i in range(n):
            rand_idx = int(rand()*n_genes)
            self.genes[s][rand_idx] += (2.0/self.n_species)*(np.random.rand()-0.5)
        max_mag = np.max(np.abs(self.genes[s]))
        if max_mag > 1:
            self.genes[s] = self.genes[s] / max_mag

    def mutate(self, n):
        for s in range(self.n_species):
            self.mutate_species(s, n)

    def species_concentrations(self):
        concentrations = np.zeros(self.n_species)
        for s in range(self.n_species):
            concentrations[s] = np.mean(self.species_lattices[s,:,:])
        return concentrations

