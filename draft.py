y_over_10 = train_dataset[train_dataset.y > 10]
res =StructureDataset(y_over_10 + train_dataset)
