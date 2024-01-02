from simulation.simulation import Simulation
from GPs.gp import GP

if __name__ == '__main__':
    simulation = Simulation(molecule_name="H2O", calc_name="EMT", super_cell=[3, 3, 3])
    simulation.run(time=100)
    gp = GP(simulation, descriptor="soap")
    gp.fit_model_energy()
    gp.plot()



