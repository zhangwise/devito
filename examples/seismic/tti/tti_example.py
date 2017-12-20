import numpy as np
from argparse import ArgumentParser

from devito import Dimension, Function, clear_cache
from devito.logger import warning
from examples.seismic import demo_model, Receiver, RickerSource
from examples.seismic.tti import AnisotropicWaveSolver

import matplotlib.pyplot as plt

def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              time_order=2, space_order=4, nbpml=10, **kwargs):

    nrec = 101
    # Two layer model for true velocity
    model = demo_model('layers-tti', shape=shape, spacing=spacing, nbpml=nbpml)
    # Derive timestepping from model spacing
    dt = model.critical_dt
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)
    time = np.linspace(t0, tn, nt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.015, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, lust below surface)
    rec = Receiver(name='nrec', grid=model.grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    return AnisotropicWaveSolver(model, source=src, receiver=rec,
                                 time_order=time_order,
                                 space_order=space_order, **kwargs)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, time_order=2, space_order=4, nbpml=10,
        kernel='centered', **kwargs):
        
    solver = tti_setup(shape, spacing, tn, time_order, space_order, nbpml, **kwargs)
    timings = np.zeros((8, 1))
    if space_order % 4 != 0:
        warning('WARNING: TTI requires a space_order that is a multiple of 4!')
    rec2, u2, v2, summary2 = solver.forward(autotune=autotune, kernel=kernel)
    timings[0] = np.sum([val for val in summary2.timings.values()])
    count=1

    for i in (1, 2, 4, 8, 16, 32, 64):
        clear_cache()
        solver = tti_setup(shape, spacing, tn, time_order, space_order, nbpml, **kwargs)

        fr = Dimension(name="fr")
        freqs = Function(name="freqs", shape=(i,), dimensions=(fr,))
        freqs.data[:] = np.linspace(2., 10., i).astype(np.float32)
        rec, u, v, summary = solver.forwardDFT(autotune=autotune, kernel=kernel, freqs=freqs, nfreqs=i)
        timings[count] = np.sum([val for val in summary.timings.values()])
        count += 1

    plt.figure()
    plt.plot([0, 1, 2, 4, 8, 16, 32, 64], timings,'-*r')
    plt.xlabel("number of frequency slices")
    plt.ylabel("Runtime (sec)")
    plt.show()
    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-to", "--time_order", default=2,
                        type=int, help="Time order of the simulation")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='centered',
                        choices=['centered', 'shifted'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", "-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DSE) mode")
    args = parser.parse_args()

    # 3D preset parameters
    if args.dim2:
        shape = (150, 150)
        spacing = (10.0, 10.0)
        tn = 750.0
    else:
        shape = (50, 50, 50)
        spacing = (10.0, 10.0, 10.0)
        tn = 250.0

    run(shape=shape, spacing=spacing, nbpml=args.nbpml, tn=tn,
        space_order=args.space_order, time_order=args.time_order,
        autotune=args.autotune, dse=args.dse, dle=args.dle, kernel=args.kernel)
