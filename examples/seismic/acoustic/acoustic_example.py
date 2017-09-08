import numpy as np
import click

from devito.logger import info
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, RickerSource, Receiver


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape, dtype=np.float32)
    out[:] = vel[:]
    nz = shape[-1]

    for a in range(5, nz-6):
        if len(shape) == 2:
            out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
        else:
            out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10

    return out


def acoustic_setup(dimensions=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., time_order=2, space_order=4, nbpml=10, **kwargs):
    nrec = dimensions[0]
    model = demo_model('layers', ratio=3, shape=dimensions,
                       spacing=spacing, nbpml=nbpml)

    # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', ndim=model.dim, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='nrec', ntime=nt, npoint=nrec, ndim=model.dim)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=time_order,
                                space_order=space_order, **kwargs)
    return solver


def acoustic_run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0),
                 tn=1000.0, time_order=2, space_order=4, nbpml=40, **kwargs):
    """ Wrapper method to execute a baseline forward operator """
    solver = acoustic_setup(dimensions=dimensions, spacing=spacing,
                            nbpml=nbpml, tn=tn, space_order=space_order,
                            time_order=time_order, **kwargs)

    info('Applying Forward with time order %d, space order %d' %
         (time_order, space_order))
    rec, u, summary = solver.forward(save=False)
    return summary.gflopss, summary.oi, summary.timings, [rec, u.data]


@click.group()
def example():
    """Example script for a set of acoustic operators."""
    pass


@example.command()
@click.option('--dimensions', default=(50, 50, 50),
              help='Number of grid points in each dimension')
@click.option('--spacing', default=(20.0, 20.0, 20.0),
              help='Grid spacing between each point in m')
@click.option('-tn', default=1000.0, type=float,
              help='Simulation time in ms')
@click.option('-to', '--time-order', type=int, default=2,
              help='Order of time discretization')
@click.option('-so', '--space-order', type=int, default=4,
              help='Order of space discretization')
@click.option('--nbpml', default=40, type=int,
              help='Number of PML layers')
def run(dimensions, spacing, tn, time_order, space_order, nbpml):
    acoustic_run(dimensions=dimensions, spacing=spacing, tn=tn,
                 time_order=time_order, space_order=space_order, nbpml=nbpml)


@example.command()
@click.option('--dimensions', default=(50, 50, 50),
              help='Number of grid points in each dimension')
@click.option('--spacing', default=(20.0, 20.0, 20.0),
              help='Grid spacing between each point in m')
@click.option('--tn', default=1000.0, type=float,
              help='Simulation time in ms')
@click.option('-to', '--time-order', type=int, default=2,
              help='Order of time discretization')
@click.option('-so', '--space-order', type=int, default=4,
              help='Order of space discretization')
@click.option('--nbpml', default=40, type=int,
              help='Number of PML layers')
def full(dimensions, spacing, tn, time_order, space_order, nbpml, **kwargs):

    solver = acoustic_setup(dimensions=dimensions, spacing=spacing,
                            nbpml=nbpml, tn=tn, space_order=space_order,
                            time_order=time_order, **kwargs)

    initial_vp = smooth10(solver.model.m.data, solver.model.shape_domain)
    dm = np.float32(initial_vp**2 - solver.model.m.data)
    info("Applying Forward")
    rec, u, summary = solver.forward(save=True)

    info("Applying Adjoint")
    solver.adjoint(rec, **kwargs)
    info("Applying Born")
    solver.born(dm, **kwargs)
    info("Applying Gradient")
    solver.gradient(rec, u, **kwargs)


if __name__ == "__main__":
    example()
