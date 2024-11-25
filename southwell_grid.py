import numpy as np
from scipy.io import savemat
from scipy.sparse import diags, csr_matrix
from scipy.ndimage import binary_dilation
import scipy.sparse
from scipy.sparse.linalg import *
import cv2

def hfli2q(sx, sy, x, y, z=None):
    # Check the number of arguments
    if not (4 <= len(locals()) <= 5):
        raise ValueError("Invalid number of input arguments.")

    Ny, Nx = sx.shape
    ValidMask = np.isfinite(sx) & np.isfinite(sy)

    # Expand in x-direction
    sxEx = np.column_stack((np.full((Ny, 1), np.nan), sx, np.full((Ny, 2), np.nan)))
    xEx = np.column_stack((np.full((Ny, 1), np.nan), x, np.full((Ny, 2), np.nan)))
    syEx = np.column_stack((np.full((Ny, 1), np.nan), sy, np.full((Ny, 2), np.nan)))
    yEx = np.column_stack((np.full((Ny, 1), np.nan), y, np.full((Ny, 2), np.nan)))

    ExpandMaskx = np.isnan(sxEx).astype(np.uint8)
    se_x = np.array([[1, 1, 0, 1, 0]], dtype=np.uint8)
    DilatedExpandMaskx = cv2.dilate(ExpandMaskx, se_x, iterations=1)
    Maskx = DilatedExpandMaskx[:, 1:-2] & ~ExpandMaskx[:, 1:-2]

    # Expand in y-direction
    sxEy = np.row_stack((np.full((1, Nx), np.nan), sx, np.full((2, Nx), np.nan)))
    xEy = np.row_stack((np.full((1, Nx), np.nan), x, np.full((2, Nx), np.nan)))
    syEy = np.row_stack((np.full((1, Nx), np.nan), sy, np.full((2, Nx), np.nan)))
    yEy = np.row_stack((np.full((1, Nx), np.nan), y, np.full((2, Nx), np.nan)))

    ExpandMasky = np.isnan(syEy).astype(np.uint8)
    se_y = np.array([[1], [1], [0], [1], [0]], dtype=np.uint8)
    DilatedExpandMasky = cv2.dilate(ExpandMasky, se_y, iterations=1)
    Masky = DilatedExpandMasky[1:-2, :] & ~ExpandMasky[1:-2, :]

    # Compose matrices Dx and Dy
    Num = Ny * Nx
    ee = np.ones(Num)
    Dx = diags([-ee, ee], [0, Ny], (Num, Num)).tocsc()
    Dy = diags([-ee, ee], [0, 1], (Num, Num)).tocsc()

    # Compose matrices Gx and Gy
    gx_x = (-1 / 13 * sxEx[:, :-3] + sxEx[:, 1:-2] + sxEx[:, 2:-1] - 1 / 13 * sxEx[:, 3:]) \
           * (xEx[:, 2:-1] - xEx[:, 1:-2]) * 13 / 24
    gx_y = (-1 / 13 * syEx[:, :-3] + syEx[:, 1:-2] + syEx[:, 2:-1] - 1 / 13 * syEx[:, 3:]) \
           * (yEx[:, 2:-1] - yEx[:, 1:-2]) * 13 / 24
    gy_x = (-1 / 13 * sxEy[:-3, :] + sxEy[1:-2, :] + sxEy[2:-1, :] - 1 / 13 * sxEy[3:, :]) \
           * (xEy[2:-1, :] - xEy[1:-2, :]) * 13 / 24
    gy_y = (-1 / 13 * syEy[:-3, :] + syEy[1:-2, :] + syEy[2:-1, :] - 1 / 13 * syEy[3:, :]) \
           * (yEy[2:-1, :] - yEy[1:-2, :]) * 13 / 24

    Gx = gx_x + gx_y
    Gy = gy_x + gy_y

    # O(h^3)
    gx3_x = (sxEx[:, 1:-2] + sxEx[:, 2:-1]) * (xEx[:, 2:-1] - xEx[:, 1:-2]) / 2
    gx3_y = (syEx[:, 1:-2] + syEx[:, 2:-1]) * (yEx[:, 2:-1] - yEx[:, 1:-2]) / 2
    gy3_x = (sxEy[1:-2, :] + sxEy[2:-1, :]) * (xEy[2:-1, :] - xEy[1:-2, :]) / 2
    gy3_y = (syEy[1:-2, :] + syEy[2:-1, :]) * (yEy[2:-1, :] - yEy[1:-2, :]) / 2

    Gx3 = gx3_x + gx3_y
    Gy3 = gy3_x + gy3_y

    # Use O(h^3) values if O(h^5) is not available
    Gx[Maskx == 1] = Gx3[Maskx == 1]
    Gy[Masky == 1] = Gy3[Masky == 1]

    # Remove NaN
    if z is None:
        # Compose D
        valid_Gx_indices = np.isfinite(Gx)
        valid_Gy_indices = np.isfinite(Gy)
        D =  scipy.sparse.vstack([
            Dx[valid_Gx_indices.flatten('F'),:],
            Dy[valid_Gy_indices.flatten('F'),:]])


        # # Compose G
        column_vector_x = np.concatenate([Gx[:, col][np.isfinite(Gx[:, col])] for col in range(Gx.shape[1])])
        column_vector_y = np.concatenate([Gy[:, col][np.isfinite(Gy[:, col])] for col in range(Gy.shape[1])])

        G = np.r_[(column_vector_x.reshape(-1, 1), column_vector_y.reshape(-1, 1))]


        # Solve "Rank deficient" for complete dataset by assuming Z(Ind)=0
        rows, cols, values = scipy.sparse.find(D)

        # 获取第一个值为 -1 的列索引
        Ind = cols[values == -1][0]
        # Ind = np.where(D[0, :] == -1)[0][0]
        D = D[:, np.arange(D.shape[1]) != Ind]  # 删除列

        Z = cg(D.T * D, D.T * G)[0]
        Z = np.insert(Z, Ind, 0)

        # G = np.where(np.isnan(G), 0, G)


    else:
        # Compose Dz
        Dz = diags(ee, 0, (Num, Num)).tocsc()

        # Compose D
        valid_Gx_indices = np.isfinite(Gx)
        valid_Gy_indices = np.isfinite(Gy)
        valid_z_indices = np.isfinite(z)
        D = csr_matrix(np.vstack((Dx[valid_Gx_indices, :].toarray(),
                                  Dy[valid_Gy_indices, :].toarray(),
                                  Dz[valid_z_indices, :].toarray())))

        # Compose G
        G = np.hstack((Gx[valid_Gx_indices], Gy[valid_Gy_indices], z[valid_z_indices]))

        # Calculate Z with least squares method
        Z = np.linalg.lstsq(D.toarray(), G, rcond=None)[0]

    # Reconstructed result
    z_hfli2q = Z.reshape(Ny, Nx,order='F')
    z_hfli2q[~ValidMask] = np.nan

    return z_hfli2q
