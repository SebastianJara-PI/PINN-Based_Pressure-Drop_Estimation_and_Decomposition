import torch


# Velocity loss
def velocity_loss(model, X_data_tensor_norm):

    x_ = X_data_tensor_norm[:,0:1]
    y_ = X_data_tensor_norm[:,1:2]
    z_ = X_data_tensor_norm[:,2:3]
    t_ = X_data_tensor_norm[:,3:4]

    Net = model(x_, y_, z_, t_)[:,0:3]

    u_pred = Net[:, 0:1]
    v_pred = Net[:, 1:2]
    w_pred = Net[:, 2:3]

    u_ref = X_data_tensor_norm[:, 4:5]
    v_ref = X_data_tensor_norm[:, 5:6]
    w_ref = X_data_tensor_norm[:, 6:7]

    loss_u = torch.mean((u_pred - u_ref)**2)
    loss_v = torch.mean((v_pred - v_ref)**2)
    loss_w = torch.mean((w_pred - w_ref)**2)

    return loss_u, loss_v, loss_w


# Navier-Stokes divergence loss
def NS_loss(model, X_colocation_tensor_norm, sigma_x, sigma_y, sigma_z):
    
    x_c = X_colocation_tensor_norm[:, 0:1]
    y_c = X_colocation_tensor_norm[:, 1:2]
    z_c = X_colocation_tensor_norm[:, 2:3]
    t_c = X_colocation_tensor_norm[:, 3:4]

    Net = model(x_c, y_c, z_c, t_c)
    u = Net[:, 0:1]
    v = Net[:, 1:2]
    w = Net[:, 2:3]

    u_x = torch.autograd.grad(u, x_c, grad_outputs=torch.ones_like(u), create_graph=True)[0] * 1/sigma_x
    v_y = torch.autograd.grad(v, y_c, grad_outputs=torch.ones_like(v), create_graph=True)[0] * 1/sigma_y
    w_z = torch.autograd.grad(w, z_c, grad_outputs=torch.ones_like(w), create_graph=True)[0] * 1/sigma_z

    e_div = u_x + v_y + w_z

    loss_e_div = torch.mean(e_div**2)
    
    return loss_e_div

# Pressure loss
def pressure_gradient_loss(
    model,
    X_tensor_norm,
    Y_grad_tensor,
    sigma_x,
    sigma_y,
    sigma_z,
):

    x_c = X_tensor_norm[:, 0:1]
    y_c = X_tensor_norm[:, 1:2]
    z_c = X_tensor_norm[:, 2:3]
    t_c = X_tensor_norm[:, 3:4]

    Net = model(x_c, y_c, z_c, t_c)

    p_acc = Net[:, 0:1]
    p_adv = Net[:, 1:2]
    p_vis = Net[:, 2:3]

    # Gradient of p_acc'
    p_acc_x = torch.autograd.grad(
        p_acc, x_c,
        grad_outputs=torch.ones_like(p_acc),
        create_graph=True,
    )[0] / sigma_x

    p_acc_y = torch.autograd.grad(
        p_acc, y_c,
        grad_outputs=torch.ones_like(p_acc),
        create_graph=True,
    )[0] / sigma_y

    p_acc_z = torch.autograd.grad(
        p_acc, z_c,
        grad_outputs=torch.ones_like(p_acc),
        create_graph=True,
    )[0] / sigma_z

    # Gradient of p_adv'
    p_adv_x = torch.autograd.grad(
        p_adv, x_c,
        grad_outputs=torch.ones_like(p_adv),
        create_graph=True,
    )[0] / sigma_x

    p_adv_y = torch.autograd.grad(
        p_adv, y_c,
        grad_outputs=torch.ones_like(p_adv),
        create_graph=True,
    )[0] / sigma_y

    p_adv_z = torch.autograd.grad(
        p_adv, z_c,
        grad_outputs=torch.ones_like(p_adv),
        create_graph=True,
    )[0] / sigma_z

    # Gradient of p_vis'
    p_vis_x = torch.autograd.grad(
        p_vis, x_c,
        grad_outputs=torch.ones_like(p_vis),
        create_graph=True,
    )[0] / sigma_x

    p_vis_y = torch.autograd.grad(
        p_vis, y_c,
        grad_outputs=torch.ones_like(p_vis),
        create_graph=True,
    )[0] / sigma_y

    p_vis_z = torch.autograd.grad(
        p_vis, z_c,
        grad_outputs=torch.ones_like(p_vis),
        create_graph=True,
    )[0] / sigma_z

    # Targets (reference gradients)
    grad_p_acc_ref = Y_grad_tensor[:, 0:3]
    grad_p_adv_ref = Y_grad_tensor[:, 3:6]
    grad_p_vis_ref = Y_grad_tensor[:, 6:9]

    grad_p_acc_pred = torch.cat(
        [p_acc_x, p_acc_y, p_acc_z],
        dim=1,
    )

    grad_p_adv_pred = torch.cat(
        [p_adv_x, p_adv_y, p_adv_z],
        dim=1,
    )

    grad_p_vis_pred = torch.cat(
        [p_vis_x, p_vis_y, p_vis_z],
        dim=1,
    )

    # Loss terms
    loss_acc_x = torch.mean((grad_p_acc_pred[:, 0:1] - grad_p_acc_ref[:, 0:1]) ** 2)
    loss_acc_y = torch.mean((grad_p_acc_pred[:, 1:2] - grad_p_acc_ref[:, 1:2]) ** 2)
    loss_acc_z = torch.mean((grad_p_acc_pred[:, 2:3] - grad_p_acc_ref[:, 2:3]) ** 2)

    loss_adv_x = torch.mean((grad_p_adv_pred[:, 0:1] - grad_p_adv_ref[:, 0:1]) ** 2)
    loss_adv_y = torch.mean((grad_p_adv_pred[:, 1:2] - grad_p_adv_ref[:, 1:2]) ** 2)
    loss_adv_z = torch.mean((grad_p_adv_pred[:, 2:3] - grad_p_adv_ref[:, 2:3]) ** 2)

    loss_vis_x = torch.mean((grad_p_vis_pred[:, 0:1] - grad_p_vis_ref[:, 0:1]) ** 2)
    loss_vis_y = torch.mean((grad_p_vis_pred[:, 1:2] - grad_p_vis_ref[:, 1:2]) ** 2)
    loss_vis_z = torch.mean((grad_p_vis_pred[:, 2:3] - grad_p_vis_ref[:, 2:3]) ** 2)

    loss_acc = loss_acc_x + loss_acc_y + loss_acc_z
    loss_adv = loss_adv_x + loss_adv_y + loss_adv_z
    loss_vis = loss_vis_x + loss_vis_y + loss_vis_z
    
    return (
        loss_acc,
        loss_adv,
        loss_vis,
    )
