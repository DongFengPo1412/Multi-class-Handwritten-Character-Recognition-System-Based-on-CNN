import shutil
import os

def revert():
    backup_path = "checkpoints/emnist_model_backup.pth"
    model_path = "checkpoints/emnist_model.pth"

    if os.path.exists(backup_path):
        try:
            shutil.copy(backup_path, model_path)
            print("[Revert Success] 成功将模型权重回退至之前的备份版本！")
            print(f"  已用 {backup_path} 覆盖了 {model_path}")
        except Exception as e:
            print(f"[Revert Error] 复制备份文件失败: {e}")
    else:
        print("[Revert Error] 未找到备份权重文件 checkpoints/emnist_model_backup.pth！")

if __name__ == "__main__":
    revert()
