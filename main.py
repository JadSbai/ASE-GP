from GPs.force_gp import ForceGP
from GPs.energy_gp import EnergyGP
from simulation.simulation import Simulation

if __name__ == '__main__':
    simulation = Simulation(molecule_name="H2O", calc_name="EMT", super_cell=[3, 3, 3])
    simulation.run(time=300)
    force_gp = ForceGP(simulation, descriptor="base")
    force_gp.fit_model()
    force_gp.plot()

    # energy_gp = EnergyGP(simulation, descriptor="soap")
    # energy_gp.fit_model()
    # energy_gp.plot()
