from simulation.simulation import Simulation

if __name__ == '__main__':
    simulation = Simulation(molecule_name="H2O", calc_name="EMT", super_cell=[3, 3, 3])
    simulation.run()


