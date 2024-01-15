import ase
from ase.calculators.dftb import Dftb
from ase.calculators.emt import EMT
from ase.collections import g2
from ase import units
from ase.build import molecule
import numpy as np
from ase.md import Langevin
from quippy.descriptors import Descriptor
from quippy.potential import Potential
from ase.io import write, Trajectory, read
from ase.visualize import view



def print_names():
    print(g2.names)


class Simulation:
    def __init__(self, molecule_name="H20", calc_name="EMT", super_cell=[3, 3, 3]):
        self.molecule = molecule(molecule_name)
        self.super_cell = super_cell
        self.calc_name = calc_name
        self.molecule_name = molecule_name
        self.dataset_file = f'datasets/{self.molecule_name}_{self.calc_name}.traj'
        if calc_name == 'EMT':
            self.calc = EMT()
        self.temperature = 150  # Kelvin
        self.system = None
        self.dynamics = None
        self.db = []
        self.energies = []
        self.train_positions = []
        self.validate_positions = []

    def calculate_soap_descriptors(self):
        cutoff = 5.0
        l_max = 6
        n_max = 12
        atom_sigma = 0.5
        n_Z = 2
        Z = "{1 8}"

        soap_descriptor = Descriptor(
            f"soap cutoff={cutoff} l_max={l_max} normalize=T n_max={n_max} atom_sigma={atom_sigma} n_Z={n_Z} Z={Z} n_species={n_Z} species_Z={Z}")
        train_soap_data = [soap_descriptor.calc(atoms)['data'] for atoms in self.train_positions]
        validate_soap_data = [soap_descriptor.calc(atoms)['data'] for atoms in self.validate_positions]
        return train_soap_data, validate_soap_data

    def calculate_distance_descriptors(self):
        cutoff = 5.0
        dist_descriptor = Descriptor(f"distance_2b Z1=1 Z2=8 cutoff={cutoff}")
        train_dist_data = [dist_descriptor.calc(atoms)['data'] for atoms in self.train_positions]
        validate_dist_data = [dist_descriptor.calc(atoms)['data'] for atoms in self.validate_positions]
        return train_dist_data, validate_dist_data

    def make_system(self, density=1.0):
        """ Generates a supercell of desired molecules with a desired density.
            Density in g/cm^3"""
        a = np.cbrt((sum(self.molecule.get_masses()) * units.m ** 3 * 1E-6) / (density * units.mol))
        self.molecule.set_cell((a, a, a))
        self.molecule.set_pbc((True, True, True))
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

    def set_dynamics(self, time_step=1, friction=0.0002):
        """ Generates a dynamics object for the given atomic system with a desired time step and temperature.
            Temperature in Kelvin"""
        self.dynamics = Langevin(self.system, time_step * units.fs, self.temperature * units.kB, friction)
        self.dynamics.set_temperature(temperature_K=self.temperature * units.kB)
        # self.dynamics.attach(self.print_energy, interval=1)
        self.dynamics.attach(self.print_status, interval=10)
        traj = Trajectory(self.dataset_file, 'w', self.system)
        self.dynamics.attach(traj.write, interval=5)

    def get_energies(self):
        train_energies = []
        for state in self.train_positions:
            state.calc = self.calc
            train_energies.append(state.get_potential_energy(force_consistent=True))
        validate_energies = []
        for state in self.validate_positions:
            state.calc = self.calc
            validate_energies.append(state.get_potential_energy(force_consistent=True))
        print(train_energies, validate_energies)
        return train_energies, validate_energies

    def get_forces(self):
        train_forces = []
        for state in self.train_positions:
            train_forces.append(state.get_forces())
        validate_forces = []
        for state in self.validate_positions:
            validate_forces.append(state.get_forces())
        print(train_forces)
        return train_forces, validate_forces

    def view_system(self):
        view(self.system, repeat=(3, 3, 3))

    def run(self, time=100):
        self.make_system()
        self.system.set_calculator(self.calc)
        self.set_dynamics()
        self.dynamics.run(time)
        out_traj = ase.io.read(self.dataset_file, ':')
        for at in out_traj:
            at.wrap()
            if 'momenta' in at.arrays: del at.arrays['momenta']
        write('train.xyz', out_traj[0::2])
        write('validate.xyz', out_traj[1::2])
        self.train_positions = read('train.xyz', ':')
        self.validate_positions = read('validate.xyz', ':')
