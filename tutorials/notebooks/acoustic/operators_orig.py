from devito import Eq, Operator, Function, TimeFunction, Inc, solve, sign
from devito.symbolics import retrieve_functions, INT, retrieve_derivatives
from examples.seismic.utils import get_ooc_config


def freesurface(model, eq):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation.

    Parameters
    ----------
    model : Model
        Physical model.
    eq : Eq
        Time-stepping stencil (time update) to mirror at the freesurface.
    """
    lhs, rhs = eq.args
    # Get vertical dimension and corresponding subdimension
    fsdomain = model.grid.subdomains['fsdomain']
    zfs = fsdomain.dimensions[-1]
    z = zfs.parent

    # Retrieve vertical derivatives
    dzs = {d for d in retrieve_derivatives(rhs) if z in d.dims}
    # Remove inner duplicate
    dzs = dzs - {d for D in dzs for d in retrieve_derivatives(D.expr) if z in d.dims}
    dzs = {d: d._eval_at(lhs).evaluate for d in dzs}

    # Finally get functions for evaluated derivatives
    funcs = {f for f in retrieve_functions(dzs.values())}

    mapper = {}
    # Antisymmetric mirror at negative indices
    # TODO: Make a proper "mirror_indices" tool function
    for f in funcs:
        zind = f.indices[-1]
        if (zind - z).as_coeff_Mul()[0] < 0:
            s = sign(zind.subs({z: zfs, z.spacing: 1}))
            mapper.update({f: s * f.subs({zind: INT(abs(zind))})})

    # Mapper for vertical derivatives
    dzmapper = {d: v.subs(mapper) for d, v in dzs.items()}

    fs_eq = [eq.func(lhs, rhs.subs(dzmapper), subdomain=fsdomain)]
    fs_eq.append(eq.func(lhs._subs(z, 0), 0, subdomain=fsdomain))

    return fs_eq


def laplacian(field, model, kernel):
    """
    Spatial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    """
    if kernel not in ['OT2', 'OT4']:
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1/model.m) if kernel == 'OT4' else 0
    return field.laplace + s**2/12 * biharmonic


def iso_stencil(field, model, kernel, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    forward : bool, optional
        Whether to propagate forward (True) or backward (False) in time.
    """
    # Forward or backward
    forward = kwargs.get('forward', True)
    # Define time step to be updated
    unext = field.forward if forward else field.backward
    udt = field.dt if forward else field.dt.T
    # Get the spacial FD
    lap = laplacian(field, model, kernel)
    # Get source
    q = kwargs.get('q', 0)
    # Define PDE and update rule
    #eq_time = solve(model.m * field.dt2 - lap - q + model.damp * udt, unext)
    alpha = model.a1*model.phi + model.a2*model.vsh + model.a3*model.sw + model.a4
    # eq_time = solve(field.dt2 - (alpha**2)*lap, unext)
    eq_time = solve(field.dt2 - (alpha**2)*field.laplace + model.damp * udt, unext)

    # Time-stepping stencil.
    eqns = [Eq(unext, eq_time, subdomain=model.grid.subdomains['physdomain'])]

    # Add free surface
    if model.fs:
        eqns.append(freesurface(model, Eq(unext, eq_time)))
    return eqns


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m
    alpha = model.a1*model.phi + model.a2*model.vsh + model.a3*model.sw + model.a4
    dswap = kwargs.get("dswap", False)

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save and not dswap else None,
                     time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec
    
    if dswap:
        kwargs.update(get_ooc_config(u, "write", **kwargs))

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, model, kernel)

    # Construct expression to inject source values
    # src_term = src.inject(field=u.forward, expr=src * s**2 / m)
    src_term = src.inject(field=u.forward, expr=src * s**2 / alpha)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    # rec_term = rec.interpolate(expr=u.forward)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m
    alpha = model.a1*model.phi + model.a2*model.vsh + model.a3*model.sw + model.a4
    
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    # Construct expression to inject receiver values
    # receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / alpha)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)
    # source_a = srca.interpolate(expr=v.backward)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m
    dswap = kwargs.get("dswap", False)

    # Gradient symbol and wavefield symbols
    # grad = Function(name='grad', grid=model.grid)
    grad1 = Function(name='grad1', grid=model.grid)
    grad2 = Function(name='grad2', grid=model.grid)
    grad3 = Function(name='grad3', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save and not dswap
                      else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = geometry.rec

    alpha = model.a1*model.phi + model.a2*model.vsh + model.a3*model.sw + model.a4
    div_phi = 2*(model.a1**2)*model.phi + 2*model.a1*model.a2*model.vsh + 2*model.a1*model.a3*model.sw + 2*model.a1*model.a4
    div_vsh = 2*(model.a2**2)*model.vsh + 2*model.a1*model.a2*model.phi + 2*model.a2*model.a3*model.sw + 2*model.a2*model.a4
    div_sw = 2*(model.a3**2)*model.sw + 2*model.a1*model.a3*model.phi + 2*model.a2*model.a3*model.vsh + 2*model.a3*model.a4
    
    if dswap:
        kwargs.update(get_ooc_config(u, "read", **kwargs))

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    if kernel == 'OT2':
        # gradient_update = Inc(grad, - u * v.dt2)
        gradient_update1 = Inc(grad1, - 2 * div_phi * alpha * u * v.dt2)
        gradient_update2 = Inc(grad2, - 2 * div_vsh * alpha * u * v.dt2)
        gradient_update3 = Inc(grad3, - 2 * div_sw * alpha * u * v.dt2)
    elif kernel == 'OT4':
        gradient_update = Inc(grad, - u * v.dt2 - s**2 / 12.0 * u.biharmonic(m**(-2)) * v)
    # Add expression for receiver injection
    # receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / alpha)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + [gradient_update1, gradient_update2, gradient_update3], subs=model.spacing_map,
                    name='Gradient', **kwargs)
    # return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map, name='Gradient', **kwargs)


def BornOperator(model, geometry, space_order=4,
                 kernel='OT2', **kwargs):
    """
    Construct an Linearized Born operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m

    # Create source and receiver symbols
    src = geometry.src
    rec = geometry.rec

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, model, kernel)
    eqn2 = iso_stencil(U, model, kernel, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)
