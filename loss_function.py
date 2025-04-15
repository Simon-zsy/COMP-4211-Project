import torch
import torch.linalg as linalg
from scipy.optimize import linear_sum_assignment
import numpy as np

def normalize_points(P, G):
    """
    归一化点集到均值 0，方差 1，防止数值溢出。
    """
    mean = G.mean(dim=1, keepdim=True)
    std = G.std(dim=1, keepdim=True) + 1e-4
    P_norm = (P - mean) / std
    G_norm = (G - mean) / std
    return P_norm, G_norm

def fit_ellipse(points):
    """
    为批量点集拟合椭圆，返回中心 (x_c, y_c), 长轴 a, 短轴 b, 旋转角度 theta。
    points: [B, 8, 2]，批量点集
    返回: (x_c, y_c, a, b, theta)，每个形状为 [B]
    """
    B = points.shape[0]
    x_c = torch.zeros(B, device=points.device, dtype=points.dtype)
    y_c = torch.zeros(B, device=points.device, dtype=points.dtype)
    a_vals = torch.ones(B, device=points.device, dtype=points.dtype)
    b_vals = torch.ones(B, device=points.device, dtype=points.dtype)
    theta = torch.zeros(B, device=points.device, dtype=points.dtype)
    
    for batch_idx in range(B):
        pts = points[batch_idx]
        
        # 检查无效点
        if torch.isnan(pts).any() or torch.isinf(pts).any():
            x_c[batch_idx] = 0.0
            y_c[batch_idx] = 0.0
            continue
        
        # 归一化
        pts_mean = pts.mean(dim=0)
        pts_std = pts.std(dim=0) + 1e-4
        pts_norm = (pts - pts_mean) / pts_std
        if torch.isnan(pts_norm).any() or torch.isinf(pts_norm).any():
            x_c[batch_idx] = pts_mean[0]
            y_c[batch_idx] = pts_mean[1]
            continue
        
        x, y = pts_norm[:, 0], pts_norm[:, 1]
        D = torch.stack([x**2, x*y, y**2, x, y, torch.ones_like(x)], dim=1)
        S = D.T @ D
        C = torch.zeros(6, 6, device=pts.device, dtype=pts.dtype)
        C[0, 2] = C[2, 0] = 2.0
        C[1, 1] = -1.0
        
        try:
            # 增大正则化
            eigenvalues, eigenvectors = linalg.eigh(S + 1e-3 * torch.eye(6, device=S.device))
            valid = (eigenvalues > 0) & torch.isfinite(eigenvalues)
            if valid.sum() == 0:
                raise ValueError("No valid eigenvalues")
            a_vec = eigenvectors[:, valid][:, 0]
            if torch.abs(a_vec.T @ C @ a_vec) > 1e-6:
                a_vec = a_vec / torch.sqrt(torch.abs(a_vec.T @ C @ a_vec) + 1e-6)
            if torch.isnan(a_vec).any():
                raise ValueError("Invalid eigenvector")
        except:
            x_c[batch_idx] = pts_mean[0]
            y_c[batch_idx] = pts_mean[1]
            a_vals[batch_idx] = pts_std[0]
            b_vals[batch_idx] = pts_std[1]
            continue
        
        A, B, C_param, D, E, F = a_vec
        denom = B**2 - 4*A*C_param + 1e-4
        if denom <= 0 or not torch.isfinite(denom):
            x_c[batch_idx] = pts_mean[0]
            y_c[batch_idx] = pts_mean[1]
            a_vals[batch_idx] = pts_std[0]
            b_vals[batch_idx] = pts_std[1]
            continue
        
        x_c[batch_idx] = ((2*C_param*D - B*E) / denom) * pts_std[0] + pts_mean[0]
        y_c[batch_idx] = ((2*A*E - B*D) / denom) * pts_std[1] + pts_mean[1]
        
        A_prime = A + C_param + torch.sqrt(torch.clamp((A - C_param)**2 + B**2, min=1e-6))
        C_prime = A + C_param - torch.sqrt(torch.clamp((A - C_param)**2 + B**2, min=1e-6))
        F_prime = F - (D**2/(4*A + 1e-4) + E**2/(4*C_param + 1e-4) - D*E*B/(4*A*C_param + 1e-4))
        
        if torch.isfinite(F_prime) and torch.isfinite(A_prime) and torch.isfinite(C_prime):
            a_vals[batch_idx] = torch.sqrt(torch.clamp(-F_prime / (A_prime + 1e-4), min=1e-4, max=1e4)) * pts_std[0]
            b_vals[batch_idx] = torch.sqrt(torch.clamp(-F_prime / (C_prime + 1e-4), min=1e-4, max=1e4)) * pts_std[1]
        else:
            a_vals[batch_idx] = pts_std[0]
            b_vals[batch_idx] = pts_std[1]
        
        theta[batch_idx] = 0.5 * torch.atan2(B, A - C_param) if torch.isfinite(B) and torch.isfinite(A - C_param) else 0.0
    
    return x_c, y_c, a_vals, b_vals, theta

def shape_loss(P, G):
    """
    计算椭圆形状损失，比较预测点和 ground truth 拟合的椭圆参数。
    """
    B = P.shape[0]
    P = P.view(B, 8, 2)
    G = G.view(B, 8, 2)
    
    x_c_P, y_c_P, a_P, b_P, theta_P = fit_ellipse(P)
    x_c_G, y_c_G, a_G, b_G, theta_G = fit_ellipse(G)
    
    # 检查无效值
    if not (torch.isfinite(x_c_P).all() and torch.isfinite(a_P).all() and torch.isfinite(theta_P).all()):
        print("Warning: Invalid ellipse parameters for P")
        x_c_P = P.mean(dim=1)[:, 0]
        y_c_P = P.mean(dim=1)[:, 1]
        a_P = torch.ones(B, device=P.device) * P.std()
        b_P = a_P
        theta_P = torch.zeros(B, device=P.device)
    
    center_diff = torch.sum((torch.stack([x_c_P, y_c_P], dim=1) - torch.stack([x_c_G, y_c_G], dim=1))**2, dim=1)
    a_diff = (a_P - a_G)**2
    b_diff = (b_P - b_G)**2
    theta_diff = torch.min(torch.abs(theta_P - theta_G), 2 * np.pi - torch.abs(theta_P - theta_G))**2
    
    loss = torch.sum(center_diff + a_diff + b_diff + theta_diff) / B
    return torch.where(torch.isfinite(loss), loss, torch.tensor(0.0, device=P.device, requires_grad=True))

def boundary_loss(P, G):
    """
    计算边界损失，确保预测点靠近 ground truth 椭圆边界（反之亦然）。
    """
    B = P.shape[0]
    P = P.view(B, 8, 2)
    G = G.view(B, 8, 2)
    
    x_c_P, y_c_P, a_P, b_P, theta_P = fit_ellipse(P)
    x_c_G, y_c_G, a_G, b_G, theta_G = fit_ellipse(G)
    
    # 检查无效参数
    if not (torch.isfinite(a_P).all() and torch.isfinite(b_P).all()):
        a_P = torch.ones(B, device=P.device) * P.std()
        b_P = a_P
    if not (torch.isfinite(a_G).all() and torch.isfinite(b_G).all()):
        a_G = torch.ones(B, device=G.device) * G.std()
        b_G = a_G
    
    loss_P = torch.zeros(B, device=P.device, dtype=P.dtype)
    for b in range(B):
        P_b = P[b] - torch.tensor([x_c_G[b], y_c_G[b]], device=P.device)
        cos_theta = torch.cos(theta_G[b])
        sin_theta = torch.sin(theta_G[b])
        x_prime = P_b[:, 0] * cos_theta + P_b[:, 1] * sin_theta
        y_prime = -P_b[:, 0] * sin_theta + P_b[:, 1] * cos_theta
        d_P = torch.abs(x_prime**2 / (a_G[b]**2 + 1e-3) + y_prime**2 / (b_G[b]**2 + 1e-3) - 1)
        loss_P[b] = torch.sum(d_P) if torch.isfinite(d_P).all() else 0.0
    
    loss_G = torch.zeros(B, device=G.device, dtype=G.dtype)
    for b in range(B):
        G_b = G[b] - torch.tensor([x_c_P[b], y_c_P[b]], device=G.device)
        cos_theta = torch.cos(theta_P[b])
        sin_theta = torch.sin(theta_P[b])
        x_prime = G_b[:, 0] * cos_theta + G_b[:, 1] * sin_theta
        y_prime = -G_b[:, 0] * sin_theta + G_b[:, 1] * cos_theta
        d_G = torch.abs(x_prime**2 / (a_P[b]**2 + 1e-3) + y_prime**2 / (b_P[b]**2 + 1e-3) - 1)
        loss_G[b] = torch.sum(d_G) if torch.isfinite(d_G).all() else 0.0
    
    loss = torch.sum(loss_P + loss_G) / B
    return torch.where(torch.isfinite(loss), loss, torch.tensor(0.0, device=P.device, requires_grad=True))

def hungarian_mse_loss(pred_points, gt_points):
    """
    使用匈牙利算法匹配 pred_points 和 gt_points，计算匹配后的 MSE 损失。
    """
    B = pred_points.size(0)
    num_points = pred_points.size(1) // 2
    
    pred = pred_points.view(B, num_points, 2)
    gt = gt_points.view(B, num_points, 2)
    
    mse = 0.0
    for b in range(B):
        pred_b = pred[b]
        gt_b = gt[b]
        
        dist_matrix = torch.norm(pred_b.unsqueeze(1) - gt_b.unsqueeze(0), dim=2)
        
        if torch.isnan(dist_matrix).any() or torch.isinf(dist_matrix).any():
            print(f"Warning: dist_matrix contains NaN or inf for batch {b}")
            mse_b = torch.mean((pred_b - gt_b) ** 2)
        else:
            try:
                dist_matrix_np = dist_matrix.cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(dist_matrix_np)
                gt_matched = gt_b[col_ind]
                mse_b = torch.mean((pred_b - gt_matched) ** 2)
            except ValueError as e:
                print(f"Hungarian algorithm failed for batch {b}: {e}")
                mse_b = torch.mean((pred_b - gt_b) ** 2)
        
        mse += mse_b
    
    mse = mse / B
    return torch.where(torch.isfinite(mse), mse, torch.tensor(0.0, device=pred_points.device, requires_grad=True))

def custom_loss(pred_points, gt_points, alpha=1.0, beta=1.0, gamma=1.0):
    if(alpha != 0):
        shape_loss_value = shape_loss(pred_points, gt_points) * alpha
    else:
        shape_loss_value = 0
    # print(f"Shape loss: {shape_loss_value.item()}")
    if(beta != 0):
        boundary_loss_value = boundary_loss(pred_points, gt_points) * beta
    else:
        boundary_loss_value = 0
    # print(f"Boundary loss: {boundary_loss_value.item()}")
    dist_loss_value = hungarian_mse_loss(pred_points, gt_points) * gamma
    total_loss = shape_loss_value + boundary_loss_value + dist_loss_value

    return total_loss, shape_loss_value, boundary_loss_value, dist_loss_value

# 示例使用
if __name__ == "__main__":
    # 模拟数据
    B = 2
    P = torch.randn(B, 16)
    G = torch.randn(B, 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P, G = P.to(device), G.to(device)
    
    # 计算损失
    L_shape = shape_loss(P, G)
    L_boundary = boundary_loss(P, G)
    print(f"Shape loss: {L_shape.item()}")
    print(f"Boundary loss: {L_boundary.item()}")
    
    # 你的匈牙利算法损失
    
    L_dist = hungarian_mse_loss(P, G)
    print(f"Distribution loss: {L_dist.item()}")
    
    # 总损失
    lambda1, lambda2, lambda3 = 1.0, 1.0, 1.0
    L_total = lambda1 * L_shape + lambda2 * L_boundary + lambda3 * L_dist
    print(f"Total loss: {L_total.item()}")