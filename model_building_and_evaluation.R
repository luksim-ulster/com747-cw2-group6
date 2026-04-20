install.packages("randomForest")
install.packages("caret")
install.packages("pROC")
install.packages("ggplot2")
install.packages("C50")
install.packages("MLmetrics")

library(randomForest)
library(caret)
library(pROC)
library(ggplot2)
library(C50)
library(MLmetrics)

set.seed(42)

# load train and test data
train_data <- read.csv("train_resume_data.csv")
test_data <- read.csv("test_resume_data.csv")

# rename factor levels to valid R variable names so caret does not complain
# 0 = Rejected, 1 = Hired
train_data$recruiter_decision <- factor(
  ifelse(train_data$recruiter_decision == 1, "Hired", "Rejected")
)
test_data$recruiter_decision <- factor(
  ifelse(test_data$recruiter_decision == 1, "Hired", "Rejected")
)


# DECISION TREE (C5.0)


control <- trainControl(
  method = "repeatedcv",
  number = 10,
  savePredictions = TRUE,
  repeats = 3,
  classProbs = TRUE  # needed for ROC later
)

custom_grid <- expand.grid(
  model = c("rules", "tree"),
  trials = c(20, 25, 30, 35, 40),
  winnow = c(FALSE)
)

dt_model <- train(
  recruiter_decision ~ .,
  method = "C5.0",
  data = train_data,
  trControl = control,
  tuneGrid = custom_grid
)

dt_predictions <- predict(dt_model, newdata = test_data)
dt_cm <- confusionMatrix(dt_predictions, test_data$recruiter_decision, positive = "Hired")
print(dt_cm)

# extract metrics - using "Hired" as the positive class
dt_accuracy <- dt_cm$overall["Accuracy"]
dt_precision <- dt_cm$byClass["Precision"]
dt_recall <- dt_cm$byClass["Recall"]
dt_f1 <- dt_cm$byClass["F1"]

# probabilities for ROC (probability of being Hired)
dt_probs <- predict(dt_model, newdata = test_data, type = "prob")[, "Hired"]


# RANDOM FOREST MODEL


rf_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  savePredictions = TRUE,
  repeats = 3,
  classProbs = TRUE
)

rf_grid <- expand.grid(mtry = c(3, 5, 7, 10))

rf_model <- train(
  recruiter_decision ~ .,
  data = train_data,
  method = "rf",
  trControl = rf_control,
  tuneGrid = rf_grid,
  ntree = 500
)

print(rf_model)

rf_predictions <- predict(rf_model, newdata = test_data)
rf_cm <- confusionMatrix(rf_predictions, test_data$recruiter_decision, positive = "Hired")
print(rf_cm)

rf_accuracy <- rf_cm$overall["Accuracy"]
rf_precision <- rf_cm$byClass["Precision"]
rf_recall <- rf_cm$byClass["Recall"]
rf_f1 <- rf_cm$byClass["F1"]

rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, "Hired"]


# PLOT 1: BAR PLOT OF ACCURACY, PRECISION, RECALL, F1


metrics_df <- data.frame(
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1"), 2),
  Value = c(dt_accuracy, dt_precision, dt_recall, dt_f1,
            rf_accuracy, rf_precision, rf_recall, rf_f1),
  Model = rep(c("Decision Tree", "Random Forest"), each = 4)
)

# fix factor order for consistent bar ordering
metrics_df$Metric <- factor(metrics_df$Metric, levels = c("Accuracy", "Precision", "Recall", "F1"))

metrics_plot <- ggplot(metrics_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.85) +
  geom_text(
    aes(label = round(Value, 3)),
    position = position_dodge(width = 0.9),
    vjust = -0.4,
    size = 3
  ) +
  scale_fill_manual(values = c("Decision Tree" = "#4C72B0", "Random Forest" = "#DD8452")) +
  ylim(0, 1.1) +
  labs(title = "Model Performance Comparison", x = "Metric", y = "Score", fill = "Model") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "top"
  )

ggsave("metrics_comparison.png", plot = metrics_plot, width = 8, height = 6, dpi = 300, bg = "white")
print(metrics_plot)


# PLOT 2: CONFUSION MATRICES


plot_confusion_matrix <- function(cm, title) {
  cm_table <- as.data.frame(cm$table)
  colnames(cm_table) <- c("Predicted", "Actual", "Count")

  ggplot(cm_table, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), size = 6) +
    scale_fill_gradient(low = "#D6E4F0", high = "#2471A3") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "right"
    )
}

dt_cm_plot <- plot_confusion_matrix(dt_cm, "Decision Tree Confusion Matrix")
ggsave("dt_confusion_matrix.png", plot = dt_cm_plot, width = 6, height = 5, dpi = 300, bg = "white")
print(dt_cm_plot)

rf_cm_plot <- plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
ggsave("rf_confusion_matrix.png", plot = rf_cm_plot, width = 6, height = 5, dpi = 300, bg = "white")
print(rf_cm_plot)


# PLOT 3: ROC AUC CURVES


# pROC expects a numeric response - convert Hired to 1, Rejected to 0
actual_numeric <- ifelse(test_data$recruiter_decision == "Hired", 1, 0)

dt_roc <- roc(actual_numeric, dt_probs, quiet = TRUE)
rf_roc <- roc(actual_numeric, rf_probs, quiet = TRUE)

dt_auc <- round(auc(dt_roc), 3)
rf_auc <- round(auc(rf_roc), 3)

dt_roc_df <- data.frame(
  FPR = 1 - dt_roc$specificities,
  TPR = dt_roc$sensitivities,
  Model = paste0("Decision Tree (AUC = ", dt_auc, ")")
)

rf_roc_df <- data.frame(
  FPR = 1 - rf_roc$specificities,
  TPR = rf_roc$sensitivities,
  Model = paste0("Random Forest (AUC = ", rf_auc, ")")
)

roc_df <- rbind(dt_roc_df, rf_roc_df)

roc_plot <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "grey50") +
  scale_color_manual(values = setNames(
    c("#4C72B0", "#DD8452"),
    c(paste0("Decision Tree (AUC = ", dt_auc, ")"), paste0("Random Forest (AUC = ", rf_auc, ")"))
  )) +
  labs(title = "ROC AUC Curves", x = "False Positive Rate", y = "True Positive Rate", color = "Model") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = c(0.65, 0.15)
  )

ggsave("roc_auc_curves.png", plot = roc_plot, width = 7, height = 6, dpi = 300, bg = "white")
print(roc_plot)


# PLOT 4: TOP FEATURE IMPORTANCES (BOTH MODELS)


dt_importance <- varImp(dt_model)$importance
dt_importance$Feature <- rownames(dt_importance)
dt_importance <- dt_importance[order(dt_importance$Overall, decreasing = TRUE), ]
dt_top15 <- head(dt_importance, 15)

dt_importance_plot <- ggplot(dt_top15, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "#4C72B0", color = "black", alpha = 0.85) +
  coord_flip() +
  labs(title = "Decision Tree: Top 15 Feature Importances", x = "Feature", y = "Importance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("dt_feature_importance.png", plot = dt_importance_plot, width = 8, height = 6, dpi = 300, bg = "white")
print(dt_importance_plot)

rf_importance <- varImp(rf_model)$importance
rf_importance$Feature <- rownames(rf_importance)
rf_importance <- rf_importance[order(rf_importance$Overall, decreasing = TRUE), ]
rf_top15 <- head(rf_importance, 15)

rf_importance_plot <- ggplot(rf_top15, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "#DD8452", color = "black", alpha = 0.85) +
  coord_flip() +
  labs(title = "Random Forest: Top 15 Feature Importances", x = "Feature", y = "Importance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("rf_feature_importance.png", plot = rf_importance_plot, width = 8, height = 6, dpi = 300, bg = "white")
print(rf_importance_plot)
