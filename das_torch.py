import torch
from torch.nn.functional import grid_sample


PI = 3.14159265359


def DAS_Beamforming(
    P,
    grid,
    ang_list=None,
    ele_list=None,
    rxfnum=2,
    dtype=torch.float,
    device=torch.device("cuda:0"),
):

    if ang_list is None:
        ang_list = range(P.angles.shape[0])
    elif not hasattr(ang_list, "__getitem__"):
        ang_list = [ang_list]
    if ele_list is None:
        ele_list = range(P.ele_pos.shape[0])
    elif not hasattr(ele_list, "__getitem__"):
        ele_list = [ele_list]

    # Convert plane wave data to tensors
    angles = torch.tensor(P.angles, dtype=dtype, device=device)
    ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
    fc = torch.tensor(P.fc, dtype=dtype, device=device)
    fs = torch.tensor(P.fs, dtype=dtype, device=device)
    fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)
    c = torch.tensor(P.c, dtype=dtype, device=device)
    time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)

    # Convert grid to tensor
    out_shape = grid.shape[:-1]
    grid = torch.tensor(grid, dtype=dtype, device=device).reshape(-1, 3)

    # Store other information as well
    ang_list = torch.tensor(ang_list, dtype=torch.long, device=device)
    ele_list = torch.tensor(ele_list, dtype=torch.long, device=device)
    dtype = dtype
    device = device

    # Load data onto device as a torch tensor
    idata, qdata = P.idata, P.qdata
    idata = torch.tensor(idata, dtype=dtype, device=device)
    qdata = torch.tensor(qdata, dtype=dtype, device=device)

    # Compute delays in meters
    nangles = len(ang_list)
    nelems = len(ele_list)
    npixels = grid.shape[0]
    xlims = (ele_pos[0, 0], ele_pos[-1, 0])  # Aperture width
    txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
    rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
    txapo = torch.ones((nangles, npixels), dtype=dtype, device=device)
    rxapo = torch.ones((nelems, npixels), dtype=dtype, device=device)
    for i, tx in enumerate(ang_list):
        txdel[i] = delay_plane(grid, angles[[tx]])
        txdel[i] += time_zero[tx] * c
        txapo[i] = apod_plane(grid, angles[tx], xlims)
    for j, rx in enumerate(ele_list):
        rxdel[j] = delay_focus(grid, ele_pos[[rx]])
        rxapo[i] = apod_focus(grid, ele_pos[rx])
    # Convert to samples
    txdel *= fs / c
    rxdel *= fs / c

    # Initialize the output array
    idas = torch.zeros(npixels, dtype=dtype, device=device)
    qdas = torch.zeros(npixels, dtype=dtype, device=device)
    # Loop over angles and elements
    for t, td, ta in zip(ang_list, txdel, txapo):
        for r, rd, ra in zip(ele_list, rxdel, rxapo):
            iq = torch.stack((idata[t, r], qdata[t, r]), dim=0).view(1, 2, 1, -1)
            delays = td + rd
            dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
            dgs = torch.cat((dgs, 0 * dgs), axis=-1)
            ifoc, qfoc = grid_sample(iq, dgs, align_corners=False).view(2, -1)
            if fdemod != 0:
                tshift = delays.view(-1) / fs - grid[:, 2] * 2 / c
                theta = 2 * PI * fdemod * tshift
                ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)
            apods = ta * ra

            idas += ifoc * apods
            qdas += qfoc * apods

    idas = idas.view(out_shape)
    qdas = qdas.view(out_shape)
    return idas, qdas


def DAS2img(
    P,
    grid,
    ang_list=None,
    ele_list=None,
    rxfnum=2,
    dtype=torch.float,
    device=torch.device("cuda:0"),
):

    if ang_list is None:
        ang_list = range(P.angles.shape[0])
    elif not hasattr(ang_list, "__getitem__"):
        ang_list = [ang_list]
    if ele_list is None:
        ele_list = range(P.ele_pos.shape[0])
    elif not hasattr(ele_list, "__getitem__"):
        ele_list = [ele_list]

    # Convert plane wave data to tensors
    angles = torch.tensor(P.angles, dtype=dtype, device=device)
    ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
    fc = torch.tensor(P.fc, dtype=dtype, device=device)
    fs = torch.tensor(P.fs, dtype=dtype, device=device)
    fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)
    c = torch.tensor(P.c, dtype=dtype, device=device)
    time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)

    # Convert grid to tensor
    out_shape = grid.shape[:-1]
    grid = torch.tensor(grid, dtype=dtype, device=device).reshape(-1, 3)

    # Store other information as well
    ang_list = torch.tensor(ang_list, dtype=torch.long, device=device)
    ele_list = torch.tensor(ele_list, dtype=torch.long, device=device)
    dtype = dtype
    device = device

    # Load data onto device as a torch tensor
    idata, qdata = P.idata, P.qdata
    idata = torch.tensor(idata, dtype=dtype, device=device)
    qdata = torch.tensor(qdata, dtype=dtype, device=device)

    # Compute delays in meters
    nangles = len(ang_list)
    nelems = len(ele_list)
    npixels = grid.shape[0]
    xlims = (ele_pos[0, 0], ele_pos[-1, 0])  # Aperture width
    txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
    rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
    txapo = torch.ones((nangles, npixels), dtype=dtype, device=device)
    rxapo = torch.ones((nelems, npixels), dtype=dtype, device=device)
    for i, tx in enumerate(ang_list):
        txdel[i] = delay_plane(grid, angles[[tx]])
        txdel[i] += time_zero[tx] * c
        txapo[i] = apod_plane(grid, angles[tx], xlims)
    for j, rx in enumerate(ele_list):
        rxdel[j] = delay_focus(grid, ele_pos[[rx]])
        rxapo[i] = apod_focus(grid, ele_pos[rx])
    # Convert to samples
    txdel *= fs / c
    rxdel *= fs / c

    # Initialize the output array
    idas = torch.zeros(npixels, dtype=dtype, device=device)
    qdas = torch.zeros(npixels, dtype=dtype, device=device)
    # Loop over angles and elements
    for t, td, ta in zip(ang_list, txdel, txapo):
        for r, rd, ra in zip(ele_list, rxdel, rxapo):
            iq = torch.stack((idata[t, r], qdata[t, r]), dim=0).view(1, 2, 1, -1)
            delays = td + rd
            dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
            dgs = torch.cat((dgs, 0 * dgs), axis=-1)
            ifoc, qfoc = grid_sample(iq, dgs, align_corners=False).view(2, -1)
            if fdemod.cpu() != 0:
                tshift = delays.view(-1) / fs - grid[:, 2] * 2 / c
                theta = 2 * PI * fdemod * tshift
                ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)
            apods = ta * ra

            idas += ifoc * apods
            qdas += qfoc * apods

    idas = idas.view(out_shape)
    qdas = qdas.view(out_shape)

    iq1 = idas + 1j * qdas  # Transpose for display purposes
    img = 20 * torch.log10(torch.abs(iq1))  # Log-compress
    img -= torch.amax(img)  # Normalize by max value
    return img


## Simple phase rotation of I and Q component by complex angle theta
def _complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr


## Compute distance to user-defined pixels from elements
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   ele_pos Element positions in x,y,z  [nelems, 3]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_focus(grid, ele_pos):
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    # Output has shape [nelems, npixels]
    return dist


## Compute distance to user-defined pixels for plane waves
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   angles  Plane wave angles (radians) [nangles]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_plane(grid, angles):
    # Use broadcasting to simplify computations
    x = grid[:, 0].unsqueeze(0)
    z = grid[:, 2].unsqueeze(0)
    # For each element, compute distance to pixels
    dist = x * torch.sin(angles) + z * torch.cos(angles)
    # Output has shape [nangles, npixels]
    return dist


## Compute rect apodization to user-defined pixels for desired f-number
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid        Pixel positions in x,y,z        [npixels, 3]
#   ele_pos     Element positions in x,y,z      [nelems, 3]
#   fnum        Desired f-number                scalar
#   min_width   Minimum width to retain         scalar
# OUTPUTS
#   apod    Apodization for each pixel to each element  [nelems, npixels]
def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
    # Get vector between elements and pixels via broadcasting
    ppos = grid.unsqueeze(0)
    epos = ele_pos.view(-1, 1, 3)
    v = ppos - epos
    # Select (ele,pix) pairs whose effective fnum is greater than fnum
    mask = torch.abs(v[:, :, 2] / v[:, :, 0]) > fnum
    mask = mask | (torch.abs(v[:, :, 0]) <= min_width)
    # Also account for edges of aperture
    mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
    mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))
    # Convert to float and normalize across elements (i.e., delay-and-"average")
    apod = mask.float()
    # apod /= torch.sum(apod, 0, keepdim=True)
    # Output has shape [nelems, npixels]
    return apod


## Compute rect apodization to user-defined pixels for desired f-number
# Retain only pixels that lie within the aperture projected along the transmit angle.
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z            [npixels, 3]
#   angles  Plane wave angles (radians)         [nangles]
#   xlims   Azimuthal limits of the aperture    [2]
# OUTPUTS
#   apod    Apodization for each angle to each element  [nangles, npixels]
def apod_plane(grid, angles, xlims):
    pix = grid.unsqueeze(0)
    ang = angles.view(-1, 1, 1)
    # Project pixels back to aperture along the defined angles
    x_proj = pix[:, :, 0] - pix[:, :, 2] * torch.tan(ang)
    # Select only pixels whose projection lie within the aperture, with fudge factor
    mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)
    # Convert to float and normalize across angles (i.e., delay-and-"average")
    apod = mask.float()
    # apod /= torch.sum(apod, 0, keepdim=True)
    # Output has shape [nangles, npixels]
    return apod
