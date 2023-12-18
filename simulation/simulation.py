from ase.calculators.emt import EMT
from ase.collections import g2
from ase import units
from ase.build import molecule
import numpy as np
from ase.md import Langevin
from quippy.potential import Potential
from ase.io import write


def print_names():
    print(g2.names)


class Simulation:
    def __init__(self, molecule_name="H20", calc_name="EMT", super_cell=[3, 3, 3]):
        self.molecule = molecule(molecule_name)
        self.super_cell = super_cell
        self.calc_name = calc_name
        self.molecule_name = molecule_name
        if calc_name == 'EMT':
            self.calc = EMT()
        elif calc_name == 'DFT':
            self.calc = Potential('TB DFTB', param_filename='dftb-params.xml')
        self.temperature = 150  # Kelvin
        self.system = None
        self.dynamics = None
        self.db = []

    def make_system(self, density=1.0):
        """ Generates a supercell of desired molecules with a desired density.
            Density in g/cm^3"""
        a = np.cbrt((sum(self.molecule.get_masses()) * units.m ** 3 * 1E-6) / (density * units.mol))
        self.molecule.set_cell((a, a, a))
        self.molecule.set_pbc((True, True, True))
        # return cp(h2o.repeat(super_cell))
        self.system = self.molecule.repeat(self.super_cell)

    def print_energy(self):
        """Function to print the potential, kinetic and total energy."""
        epot = self.system.get_potential_energy() / len(self.system)
        ekin = self.system.get_kinetic_energy() / len(self.system)
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

    def print_status(self):
        print('Step = {}, time = {} [fs], T = {} [K]'.format(
            self.dynamics.nsteps,
            self.dynamics.nsteps * self.dynamics.dt / units.fs,
            self.system.get_kinetic_energy() / (1.5 * units.kB * len(self.system))
        ))

    def collect_data(self):
        self.db.append(self.system.copy())

    def set_dynamics(self, time_step=1, friction=0.002):
        """ Generates a dynamics object for the given atomic system with a desired time step and temperature.
            Temperature in Kelvin"""
        self.dynamics = Langevin(self.system, time_step * units.fs, self.temperature * units.kB, friction)
        self.dynamics.set_temperature(temperature_K=self.temperature * units.kB)
        self.dynamics.attach(self.print_energy, interval=1)
        self.dynamics.attach(self.print_status, interval=10)
        self.dynamics.attach(self.collect_data, interval=10)

    def run(self, time=100):
        self.make_system()
        self.system.set_calculator(self.calc)
        self.set_dynamics()
        self.dynamics.run(time)
        write(f'/datasets/{self.molecule_name}_{self.calc_name}_db.xyz', self.db)
