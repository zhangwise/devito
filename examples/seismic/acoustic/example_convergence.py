import numpy as np
import click

from devito import info, error, clear_cache
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, RickerSource, Receiver
try:
    from opescibench import Executor, Benchmark
except:
    error('Could not find the Opesci benchmarking utility.\n'
          'Please install this from https://github.com/opesci/opescibench')
    raise ImportError('Missing opescibench utility package')


@click.group()
def example():
    """
    Convergence test and benchmark script for acoustic forward
    operators.
    """
    pass


@example.command()
@click.option('-d', '--dim', type=int, default=2,
              help='Number of spatial dimensions: 2D or 3D')
@click.option('--tn', default=300.0, type=float,
              help='Simulation time in ms')
@click.option('--ref-shape', type=int, default=1680,
              help='Grid size of the reference solution')
@click.option('--ref-order', type=int, default=40,
              help='Spatial order of the reference solution')
@click.option('--ref-data', type=click.Choice(['exec', 'load', 'store']),
              default='exec', help='What to do with reference solution data')
@click.option('-s', '--scale', type=int, default=(1, ), multiple=True,
              help='Factor by which to scale the reference spacing/shape')
@click.option('-o', '--order', type=int, default=(2, ), multiple=True,
              help='Spatial discretization order to run experiment')
def bench(dim, tn, ref_shape, ref_order, ref_data, scale, order):
    """ Convergence benchmark for an acoustic forward operator """

    # Reference run at with constant velocity and 1km grid in each dimension
    shape = tuple(ref_shape for _ in range(dim))
    spacing = tuple(1000. / i for i in shape)
    model = demo_model('layers', vp_top=1.5, vp_bottom=1.5,
                       shape=shape, spacing=spacing, nbpml=0)

    # Derive timestepping from model spacing
    t0 = 0.0
    nt = int(1 + (tn-t0) / model.critical_dt)  # Number of timesteps
    time = np.linspace(t0, tn, nt)  # Discretized time axis
    tidx = (nt + 2) % 3  # Final time index in wavefield buffer

    # Define source in center of domain
    src = RickerSource(name='src', ndim=model.dim, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5

    # Define receivers in center of domain, but spread across x
    nrec = 20
    rec = Receiver(name='rec', ntime=nt, npoint=nrec, ndim=model.dim)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Generate/load/store the reference solution
    fname = 'uref_dim%s_shape%s_order%s_tn%s' % (dim, ref_shape, ref_order, tn)
    if ref_data in ['exec', 'store']:
        # Run reference model and store store final wavefield
        info('Computing reference solution with shape %s order %d' %
             (shape, ref_order))
        solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                    space_order=ref_order)
        u_r = solver.forward(save=False)[1]
        u_ref = u_r.data[tidx].copy()

        if ref_data in ['store']:
            info('Storing reference solution: %s' % fname)
            np.save(fname, u_ref)

    elif ref_data in ['load']:
        info('Loading reference solution: %s.npy' % fname)
        u_ref = np.load('%s.npy' % fname)
        u_ref.reshape(shape)

    parameters = {
        'dim': dim, 'tn': tn, 'scale': list(scale), 'order': list(order),
        'ref_shape': ref_shape, 'ref_order': ref_order,
    }

    class AcousticForward(Executor):
        """Executor for an acoustic forward operator"""

        def run(self, dim, tn, scale, order, ref_shape, **kwargs):
            # from IPython import embed; embed()

            # Create model for experiment
            shape = tuple(ref_shape / scale for _ in range(dim))
            spacing = tuple(1000. / i for i in shape)
            m0 = demo_model('layers', vp_top=1.5, vp_bottom=1.5,
                            shape=shape, spacing=spacing, nbpml=0)

            # Run experiment with specified grid scaling and order
            solver = AcousticWaveSolver(m0, source=src, receiver=rec,
                                        space_order=order)
            _, u, timings = solver.forward(save=False)
            u0 = u.data[tidx]

            # Compute and log resulting error with respect to reference
            u_r = u_ref[::scale, ::scale]
            error = np.linalg.norm(
                u0.reshape(-1) / np.linalg.norm(u0.reshape(-1))
                - u_r.reshape(-1) / np.linalg.norm(u_r.reshape(-1))
            )
            info("Error[order: %d, scale: %d] %s" % (order, scale, error))

            # Register timings and error measurements
            self.register(timings['main'].time, measure='time')
            self.register(error, measure='error')

            clear_cache()

    benchmark = Benchmark(name='acoustic', parameters=parameters)
    benchmark.execute(AcousticForward(), warmups=0, repeats=3)
    benchmark.save()


if __name__ == "__main__":
    example()
