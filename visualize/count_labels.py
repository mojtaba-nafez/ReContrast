

def count_unique_labels_of_dataset(dataset, dataset_name):
    label_counts = {}

    # Enumerate through the train_dataset
    for i, (data, label) in enumerate(dataset):
        # Count the occurrences of each label
        label_counts[label] = label_counts.get(label, 0) + 1

    # Print the count of unique labels
    print(f"\nCount of Unique Labels of {dataset_name}:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
