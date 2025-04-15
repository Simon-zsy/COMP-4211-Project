import torch
import torch.nn as nn
import torch.optim as optim
from network import ConvNeXtUNet
from data_loader import get_dataloaders
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment
from loss_function import custom_loss


# 训练和验证函数
def train_model(model, train_loader, val_loader, optimizer, num_epochs=200, unfreeze_epoch=20, accum_steps=1, save_interval=10, pretrain_epochs=10):
    writer = SummaryWriter('runs/experiment')
    
    best_val_loss = float('inf')  # 跟踪最佳验证损失
    save_dir = 'model1'  # 模型保存目录
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs + pretrain_epochs,
        eta_min=1e-6  # 最小学习率
    )

    # 预训练阶段，仅使用MSE Loss
    if pretrain_epochs > 0:
        print(f"Starting pretraining with MSE loss for {pretrain_epochs} epochs...")
        for epoch in range(pretrain_epochs):
            model.train()
            train_mse = 0.0

            optimizer.zero_grad()

            if epoch == unfreeze_epoch:
                print("Unfreezing backbone weights...")
                base_model = model.module if isinstance(model, nn.DataParallel) else model
                for param in base_model.encoder.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            for i, (images, gt_points) in enumerate(train_loader):
                images = images.cuda()
                gt_points = gt_points.cuda()
                
                pred_points = model(images)
                
                # 使用自定义损失函数，但只启用MSE部分
                _, _, _, mse_loss = custom_loss(
                    pred_points, gt_points, alpha=0, beta=0, gamma=1.0
                )
                
                mse_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_mse += mse_loss.item() * images.size(0)
            
            train_mse = train_mse / len(train_loader.dataset)
            
            # 验证
            model.eval()
            val_mse = 0.0
            with torch.no_grad():
                for images, gt_points in val_loader:
                    images = images.cuda()
                    gt_points = gt_points.cuda()
                    pred_points = model(images)
                    
                    _, _, _, mse_loss = custom_loss(
                        pred_points, gt_points, alpha=0.0, beta=0.0, gamma=1.0
                    )
                    
                    val_mse += mse_loss.item() * images.size(0)
            
            val_mse = val_mse / len(val_loader.dataset)
            
            writer.add_scalar('Pretrain/train_mse', train_mse, epoch)
            writer.add_scalar('Pretrain/val_mse', val_mse, epoch)
            
            print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
            
            scheduler.step()
            
            # 每间隔save_interval保存模型
            if (epoch + 1) % save_interval == 0:
                save_path = os.path.join(save_dir, f'convnext_unet_pretrain_epoch_{epoch+1}.pth')
                base_model = model.module if isinstance(model, nn.DataParallel) else model
                try:
                    torch.save(base_model.state_dict(), save_path)
                    print(f"Saved pretrain model checkpoint at {save_path}")
                except OSError as e:
                    print(f"WARNING: Failed to save pretrain checkpoint due to: {e}")
        
        print("Pretraining completed. Starting full training with all loss components...")

    # 主训练阶段，使用所有损失组件
    for epoch in range(pretrain_epochs, num_epochs):
        model.train()
        train_total_loss = 0.0
        train_shape_loss = 0.0
        train_boundary_loss = 0.0
        train_mse_loss = 0.0
        
        optimizer.zero_grad()
        
        if epoch == unfreeze_epoch:
            print("Unfreezing backbone weights...")
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            for param in base_model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for i, (images, gt_points) in enumerate(train_loader):
            images = images.cuda()
            gt_points = gt_points.cuda()
            
            pred_points = model(images)
            
            total_loss, shape_loss_value, boundary_loss_value, mse_loss = custom_loss(
                pred_points, gt_points, alpha=0.01, beta=1e-4, gamma=5.0
            )
            total_loss = total_loss / accum_steps
            total_loss.backward()
            
            # 添加梯度裁剪以提高稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            train_total_loss += total_loss.item() * images.size(0) * accum_steps
            train_shape_loss += shape_loss_value.item() * images.size(0)
            train_boundary_loss += boundary_loss_value.item() * images.size(0)
            train_mse_loss += mse_loss.item() * images.size(0)
            
            # if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        
        train_total_loss = train_total_loss / len(train_loader.dataset)
        train_shape_loss = train_shape_loss / len(train_loader.dataset)
        train_boundary_loss = train_boundary_loss / len(train_loader.dataset)
        train_mse_loss = train_mse_loss / len(train_loader.dataset)
        
        model.eval()
        val_total_loss = 0.0
        val_shape_loss = 0.0
        val_boundary_loss = 0.0
        val_mse_loss = 0.0
        
        with torch.no_grad():
            for images, gt_points in val_loader:
                images = images.cuda()
                gt_points = gt_points.cuda()
                pred_points = model(images)
                total_loss, shape_loss_value, boundary_loss_value, mse_loss = custom_loss(
                    pred_points, gt_points, alpha=0.01, beta=1e-4, gamma=5.0
                )
                val_total_loss += total_loss.item() * images.size(0)
                val_shape_loss += shape_loss_value.item() * images.size(0)
                val_boundary_loss += boundary_loss_value.item() * images.size(0)
                val_mse_loss += mse_loss.item() * images.size(0)
        
        val_total_loss = val_total_loss / len(val_loader.dataset)
        val_shape_loss = val_shape_loss / len(val_loader.dataset)
        val_boundary_loss = val_boundary_loss / len(val_loader.dataset)
        val_mse_loss = val_mse_loss / len(val_loader.dataset)
        
        # 主训练阶段的epoch计数器从0开始
        writer.add_scalar('Loss/train_total', train_total_loss, epoch)
        writer.add_scalar('Loss/val_total', val_total_loss, epoch)
        writer.add_scalar('Loss/train_shape', train_shape_loss, epoch)
        writer.add_scalar('Loss/val_shape', val_shape_loss, epoch)
        writer.add_scalar('Loss/train_boundary', train_boundary_loss, epoch)
        writer.add_scalar('Loss/val_boundary', val_boundary_loss, epoch)
        writer.add_scalar('Loss/train_mse', train_mse_loss, epoch)
        writer.add_scalar('Loss/val_mse', val_mse_loss, epoch)
        
        print(f"Main Epoch {epoch+1}/{num_epochs}, "
              f"Train Total: {train_total_loss:.6f}, Shape: {train_shape_loss:.6f}, "
              f"Boundary: {train_boundary_loss:.6f}, MSE: {train_mse_loss:.6f}, "
              f"Val Total: {val_total_loss:.6f}, Shape: {val_shape_loss:.6f}, "
              f"Boundary: {val_boundary_loss:.6f}, MSE: {val_mse_loss:.6f}")
        
        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'convnext_unet_epoch_{epoch+1}.pth')
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            try:
                torch.save(base_model.state_dict(), save_path)
                print(f"Saved model checkpoint at {save_path}")
            except OSError as e:
                print(f"WARNING: Failed to save checkpoint due to: {e}")
                try:
                    # 尝试保存为半精度
                    state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                                for k, v in base_model.state_dict().items()}
                    torch.save(state_dict, save_path)
                    print(f"Saved half-precision model checkpoint at {save_path}")
                except Exception as e2:
                    print(f"Could not save model: {e2}")
        
        # 更新最佳模型
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_path = os.path.join(save_dir, 'convnext_unet_best.pth')
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            try:
                torch.save(base_model.state_dict(), save_path)
                print(f"Updated best model with val_total_loss={val_total_loss:.6f} at {save_path}")
            except OSError as e:
                print(f"WARNING: Failed to save best model due to: {e}")
                try:
                    state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                                for k, v in base_model.state_dict().items()}
                    torch.save(state_dict, save_path)
                    print(f"Saved half-precision best model at {save_path}")
                except Exception as e2:
                    print(f"Could not save best model: {e2}")
    
    writer.close()
    return optimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXtUNet with specified GPUs")
    parser.add_argument('--device', type=str, default='0', help='GPU IDs to use (e.g., "6" or "6,7")')
    parser.add_argument('--data_root', type=str, default='/localdata/szhoubx/rm/data/dataset', help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Base batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of main training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Number of pretraining epochs with MSE loss only')
    parser.add_argument('--unfreeze_epoch', type=int, default=20, help='Epoch to unfreeze backbone')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--save_interval', type=int, default=20, help='Save model every N epochs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device_ids = [int(gpu) for gpu in args.device.split(',')]
    num_gpus = len(device_ids)
    print(f"Using GPUs: {device_ids}, Number of GPUs: {num_gpus}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_gpus=num_gpus
    )
    
    model = ConvNeXtUNet(num_points=8, pretrained=True)
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=range(num_gpus))
        print(f"Using DataParallel with GPUs: {device_ids}")
    model = model.to(device)
    
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    for param in base_model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    
    optimizer = train_model(
        model, train_loader, val_loader, optimizer,
        num_epochs=args.num_epochs,
        pretrain_epochs=args.pretrain_epochs,
        unfreeze_epoch=args.unfreeze_epoch,
        accum_steps=args.accum_steps,
        save_interval=args.save_interval
    )
    
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    try:
        torch.save(base_model.state_dict(), "model1/convnext_unet_final.pth")
        print("Saved final model successfully")
    except OSError as e:
        print(f"WARNING: Failed to save final model due to: {e}")
        try:
            # 尝试保存为半精度
            state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in base_model.state_dict().items()}
            torch.save(state_dict, "model1/convnext_unet_final_half.pth")
            print("Saved half-precision final model successfully")
        except Exception as e2:
            print(f"Could not save final model: {e2}")