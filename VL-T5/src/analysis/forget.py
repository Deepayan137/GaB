import pandas as pd



def calculate_average_forgetting(data):
    Incre_avg_forget = [0]  # Starting with no forgetting when no tasks have been trained

    # Iterate over each training session, starting from the second session (index 1)
    for t in range(1, len(data)):
        # Select the results up to and including the t-th task
        results_now = data.iloc[:, :t + 1]
        t_forget = []  # List to hold forgetting for each task at this stage
        
        # Calculate forgetting for each task up to the current one, excluding the last trained task
        for idx in range(len(results_now.columns) - 1):
            task_list = results_now.iloc[:-1, idx]  # All but the last training's accuracy for the current task
            final = results_now.iloc[-1, idx]  # The last training session's accuracy for the current task
            pre_max = max(task_list)
            # Calculate forgetting, but handle cases where no valid max was found due to missing data (-1 in your case)
            if pre_max == -1:
                t_forget.append(0)
            else:
                t_forget.append(pre_max - final)  # Forgetting as reduction from previous max accuracy
        # Calculate the average forgetting for this session
        if t_forget:
            Avg_forget = sum(t_forget) / len(t_forget)
        else:
            Avg_forget = 0  # Handle cases where there may be no valid data to compute forgetting
        Incre_avg_forget.append(Avg_forget)
    
    return Incre_avg_forget[-1]

# Load the CSV file
csv_path = 'acc_metrics/naiveblip_sgvqa_cluster_balanced_rolak_5k.csv'  # Change to your actual CSV file path
data = pd.read_csv(csv_path)
data = data.iloc[:, 1:]
# import pdb;pdb.set_trace()
# data = data.T
# if len(data) != 10 or 'q_recognition' not in data.columns:
#     q_rec = {'q_recognition': 34.75, 'q_location': 20.54, 'q_judge': 48.23, 'q_commonsense': 56.27, 'q_count': 19.64,
#              'q_action': 41.03, 'q_color': 53.29, 'q_type': 31.64, 'q_subcategory': 41.35, 'q_causal': 3.91}
#     q_rec_df = pd.DataFrame([q_rec])  # Convert dictionary to DataFrame
#     data = pd.concat([q_rec_df, data], ignore_index=True)  # Concatenate DataFrame to the existing data

# Calculate the average forgetting
average_forgetting = calculate_average_forgetting(data)
print("Average Forgetting:", average_forgetting)
