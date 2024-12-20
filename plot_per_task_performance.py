import os
import torch
import matplotlib.pyplot as plt

task_name_task_index = {
    '1back_category': 3,
    '1back_identity': 2,
    '1back_position': 1,
    '2back_category': 6,
    '2back_identity': 5,
    '2back_position': 4,
    '3back_category': 9,
    '3back_identity': 8,
    '3back_position': 7,
    'dms_category': 43,
    'dms_identity': 42,
    'dms_position': 41,
    'interdms_AABB_category_category': 18,
    'interdms_AABB_category_identity': 17,
    'interdms_AABB_category_position': 16,
    'interdms_AABB_identity_category': 15,
    'interdms_AABB_identity_identity': 14,
    'interdms_AABB_identity_position': 13,
    'interdms_AABB_position_category': 12,
    'interdms_AABB_position_identity': 11,
    'interdms_AABB_position_position': 10, 
    'interdms_ABAB_category_category': 27,
    'interdms_ABAB_category_identity': 26,
    'interdms_ABAB_category_position': 25,
    'interdms_ABAB_identity_category': 24,
    'interdms_ABAB_identity_identity': 23,
    'interdms_ABAB_identity_position': 22,
    'interdms_ABAB_position_category': 21,
    'interdms_ABAB_position_identity': 20,
    'interdms_ABAB_position_position': 19,
    'interdms_ABBA_category_category': 36,
    'interdms_ABBA_category_identity': 35,
    'interdms_ABBA_category_position': 34,
    'interdms_ABBA_identity_category': 33,
    'interdms_ABBA_identity_identity': 32,
    'interdms_ABBA_identity_position': 31,
    'interdms_ABBA_position_category': 30,
    'interdms_ABBA_position_identity': 29,
    'interdms_ABBA_position_position': 28,

    'ctxdms_category_identity_position': 40,
    'ctxdms_position_category_identity': 37,
    'ctxdms_position_identity_category': 38,
    'ctxdms_identity_position_category': 39

}

results_dir = "/home/xiaoxuan/projects/WM-abstract/results"

task_val_accuracies = torch.load( os.path.join(results_dir, f'per_task_validation_trajectories_rep5.pth'))
task_train_accuracies = torch.load( os.path.join(results_dir, f'per_task_training_trajectories_rep5.pth'))

task_names = list(task_val_accuracies.keys())


for task_id, task_name in enumerate(task_names):
    plt.figure(figsize=(24, 16))
    task_id = task_name # todo: make the aligned with the task_id
    # Plot validation accuracy per task
    plt.plot(task_val_accuracies[int(task_id)], label=f'{task_name} - Validation Accuracy')
    plt.plot(task_train_accuracies[int(task_id)], label=f'{task_name} - training Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{task_name} Accuracy')
    plt.legend(loc = "upper left", bbox_to_anchor=(1,1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{task_name}_train_val_trajectories_rep5.png'))
    