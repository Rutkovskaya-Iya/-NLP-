install.packages(c("tidyverse", "tidytext", "tm", "wordcloud", "wordcloud2",
                   "ggplot2", "ggcorrplot", "caret", "e1071", "randomForest",
                   "syuzhet", "textdata", "reshape2", "scales",
                   "RColorBrewer", "lubridate", "stringr", "knitr"))

library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud)
library(wordcloud2)
library(ggplot2)
library(ggcorrplot)
library(caret)
library(e1071)
library(randomForest)
library(syuzhet)
library(reshape2)
library(scales)
library(RColorBrewer)
library(lubridate)
library(stringr)

df_raw <- read.csv("C:/Users/ИЯ/Desktop/R Laba/1 laba/amazon_review.csv", stringsAsFactors = FALSE, 
                   encoding = "UTF-8")

cat("Размерность датасета:", nrow(df_raw), "строк,", ncol(df_raw), "столбцов\n")
cat("Столбцы:", paste(names(df_raw), collapse = ", "), "\n")

glimpse(df_raw)
summary(df_raw)


df <- df_raw %>%
  filter(!is.na(reviewText), reviewText != "") %>%       # удаляем пустые отзывы
  filter(!is.na(overall)) %>%                             # удаляем NA в оценке
  distinct(reviewText, .keep_all = TRUE) %>%              # удаляем дубликаты
  mutate(
    sentiment = case_when(
      overall <= 2 ~ "Negative",
      overall == 3 ~ "Neutral",
      overall >= 4 ~ "Positive"
    ),
    sentiment = factor(sentiment, levels = c("Negative", "Neutral", "Positive")),
    review_length   = nchar(reviewText),                  # длина отзыва (символы)
    word_count      = str_count(reviewText, "\\S+"),      # количество слов
    has_exclamation = str_detect(reviewText, "!"),        # наличие восклицания
    reviewTime = ymd(reviewTime),                    # парсинг даты
    review_year     = year(reviewTime)
  )

cat("\nПосле предобработки:", nrow(df), "строк\n")
cat("Удалено строк:", nrow(df_raw) - nrow(df), "\n\n")

cat(" Распределение тональности \n")
print(table(df$sentiment))
print(prop.table(table(df$sentiment)) * 100)

cat("\n Описательная статистика числовых признаков \n")
df %>%
  select(overall, review_length, word_count, helpful_yes, total_vote) %>%
  summary() %>%
  print()

# Функция очистки текста
clean_text <- function(text) {
  text <- tolower(text)                              # нижний регистр
  text <- str_remove_all(text, "http\\S+|www\\S+")  # удаление URL
  text <- str_remove_all(text, "[^a-z\\s]")         # только буквы
  text <- str_squish(text)                           # лишние пробелы
  return(text)
}

df <- df %>%
  mutate(clean_review = clean_text(reviewText))

# Токенизация и удаление стоп-слов
tokens <- df %>%
  select(sentiment, clean_review) %>%
  unnest_tokens(word, clean_review) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2)

cat("\nВсего токенов после очистки:", nrow(tokens), "\n")
cat("Уникальных слов:", n_distinct(tokens$word), "\n")

p1 <- ggplot(df, aes(x = sentiment, fill = sentiment)) +
  geom_bar(color = "white", width = 0.6) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 4.5) +
  scale_fill_manual(values = c("Negative" = "#e74c3c",
                               "Neutral"  = "#f39c12",
                               "Positive" = "#2ecc71")) +
  labs(title = "Рис. 1. Распределение отзывов по тональности",
       x = "Тональность", y = "Количество отзывов") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))

print(p1)
ggsave("plot1_sentiment_distribution.png", p1, width = 7, height = 5, dpi = 150)

p2 <- ggplot(df, aes(x = overall, fill = sentiment)) +
  geom_bar(color = "white") +
  scale_fill_manual(values = c("Negative" = "#e74c3c",
                               "Neutral"  = "#f39c12",
                               "Positive" = "#2ecc71")) +
  scale_x_continuous(breaks = 1:5) +
  labs(title = "Рис. 2. Распределение оценок пользователей",
       x = "Оценка (звёзды)", y = "Количество", fill = "Тональность") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(p2)
ggsave("plot2_rating_distribution.png", p2, width = 8, height = 5, dpi = 150)

p3 <- ggplot(df %>% filter(word_count < 500), 
             aes(x = sentiment, y = word_count, fill = sentiment)) +
  geom_boxplot(outlier.alpha = 0.3, color = "gray40") +
  scale_fill_manual(values = c("Negative" = "#e74c3c",
                               "Neutral"  = "#f39c12",
                               "Positive" = "#2ecc71")) +
  labs(title = "Рис. 3. Длина отзыва (в словах) по группам тональности",
       x = "Тональность", y = "Количество слов") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))

print(p3)
ggsave("plot3_length_boxplot.png", p3, width = 7, height = 5, dpi = 150)

top_words <- tokens %>%
  group_by(sentiment, word) %>%
  count(sort = TRUE) %>%
  group_by(sentiment) %>%
  slice_max(n, n = 20)

p4 <- ggplot(top_words, aes(x = reorder_within(word, n, sentiment), 
                            y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free") +
  scale_x_reordered() +
  scale_fill_manual(values = c("Negative" = "#e74c3c",
                               "Neutral"  = "#f39c12",
                               "Positive" = "#2ecc71")) +
  coord_flip() +
  labs(title = "Рис. 4. Топ-20 слов по каждой группе тональности",
       x = NULL, y = "Частота") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

print(p4)
ggsave("plot4_top_words.png", p4, width = 12, height = 6, dpi = 150)

png("plot5_wordcloud_positive.png", width = 800, height = 600)
positive_words <- tokens %>%
  filter(sentiment == "Positive") %>%
  count(word, sort = TRUE)
wordcloud(words = positive_words$word,
          freq  = positive_words$n,
          max.words = 100,
          colors = brewer.pal(8, "Greens"),
          random.order = FALSE,
          scale = c(4, 0.5))
title("Рис. 5. Облако слов: Positive-отзывы")
dev.off()
cat("Облако слов сохранено.\n")

num_vars <- df %>%
  select(overall, review_length, word_count, helpful_yes, total_vote, day_diff)

cor_matrix <- cor(num_vars, use = "complete.obs")

p6 <- ggcorrplot(cor_matrix,
                 method = "circle",
                 type = "lower",
                 lab = TRUE,
                 lab_size = 3,
                 colors = c("#e74c3c", "white", "#2ecc71"),
                 title = "Рис. 6. Матрица корреляций числовых признаков") +
  theme(plot.title = element_text(face = "bold", size = 13))

print(p6)
ggsave("plot6_correlation_matrix.png", p6, width = 7, height = 6, dpi = 150)

p7 <- ggplot(df %>% filter(word_count < 600),
             aes(x = word_count, y = overall, color = sentiment)) +
  geom_jitter(alpha = 0.25, size = 1.2, width = 0, height = 0.15) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.2) +
  scale_color_manual(values = c("Negative" = "#e74c3c",
                                "Neutral"  = "#f39c12",
                                "Positive" = "#2ecc71")) +
  labs(title = "Рис. 7. Связь длины отзыва и оценки",
       x = "Количество слов", y = "Оценка", color = "Тональность") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(p7)
ggsave("plot7_scatter_words_rating.png", p7, width = 8, height = 5, dpi = 150)

# Получаем AFINN-оценки для каждого слова
afinn <- get_sentiments("afinn")

# Средний AFINN-скор по тональности
afinn_scores <- tokens %>%
  inner_join(afinn, by = "word") %>%
  group_by(sentiment) %>%
  summarise(
    mean_score = mean(value),
    median_score = median(value),
    n_words = n()
  )

cat("\n AFINN-скоры по группам тональности \n")
print(afinn_scores)

# Визуализация AFINN-скора на уровне отзыва
review_afinn <- df %>%
  select(sentiment, clean_review) %>%
  mutate(row_id = row_number()) %>%
  unnest_tokens(word, clean_review) %>%
  inner_join(afinn, by = "word") %>%
  group_by(row_id, sentiment) %>%
  summarise(afinn_score = sum(value), .groups = "drop")

p8 <- ggplot(review_afinn, aes(x = sentiment, y = afinn_score, fill = sentiment)) +
  geom_violin(alpha = 0.7, color = "gray50") +
  geom_boxplot(width = 0.1, fill = "white", color = "gray40", outlier.alpha = 0.2) +
  scale_fill_manual(values = c("Negative" = "#e74c3c",
                               "Neutral"  = "#f39c12",
                               "Positive" = "#2ecc71")) +
  labs(title = "Рис. 8. Распределение AFINN-скора тональности по группам",
       x = "Тональность", y = "Суммарный AFINN-скор отзыва") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))

print(p8)
ggsave("plot8_afinn_violin.png", p8, width = 7, height = 5, dpi = 150)

cat("\n СТАТИСТИЧЕСКИЕ ГИПОТЕЗЫ \n\n")

# Гипотеза 1
cat("H0: Средняя длина отзыва одинакова для всех групп тональности\n")
cat("H1: Хотя бы одна группа отличается\n")

anova_result <- aov(word_count ~ sentiment, data = df %>% filter(word_count < 1000))
anova_summary <- summary(anova_result)
print(anova_summary)

p_anova <- anova_summary[[1]][["Pr(>F)"]][1]
cat("p-value ANOVA:", p_anova, "\n")
cat("Вывод:", ifelse(p_anova < 0.05, "H0 отвергается — длина значимо различается.", 
                     "H0 не отвергается."), "\n\n")

# Попарный тест Тьюки
tukey_result <- TukeyHSD(anova_result)
print(tukey_result)

# Гипотеза 2
cat("\nH0: Наличие восклицательного знака не связано с тональностью\n")
cat("H1: Есть статистически значимая связь\n")

chi_table <- table(df$has_exclamation, df$sentiment)
chi_result <- chisq.test(chi_table)
print(chi_result)
cat("p-value χ²:", chi_result$p.value, "\n")
cat("Вывод:", ifelse(chi_result$p.value < 0.05, 
                     "H0 отвергается — связь существует.", 
                     "H0 не отвергается."), "\n\n")

# Гипотеза 3
cat("H0: Распределение AFINN-скора одинаково для Positive и Negative\n")
cat("H1: AFINN-скор значимо выше для Positive\n")

pos_scores <- review_afinn %>% filter(sentiment == "Positive") %>% pull(afinn_score)
neg_scores <- review_afinn %>% filter(sentiment == "Negative") %>% pull(afinn_score)

wilcox_result <- wilcox.test(pos_scores, neg_scores, alternative = "greater")
print(wilcox_result)
cat("p-value U-тест:", wilcox_result$p.value, "\n")
cat("Вывод:", ifelse(wilcox_result$p.value < 0.05, 
                     "H0 отвергается — Positive-отзывы значимо позитивнее.", 
                     "H0 не отвергается."), "\n\n")

cat(" ПОСТРОЕНИЕ МОДЕЛЕЙ \n\n")

# Подготовка признаков
tfidf_matrix <- df %>%
  mutate(row_id = row_number()) %>%
  filter(sentiment != "Neutral") %>%
  unnest_tokens(word, clean_review) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2) %>%
  count(row_id, word) %>%
  bind_tf_idf(word, row_id, n) %>%
  group_by(row_id) %>%
  slice_max(tf_idf, n = 50) %>%   # топ-50 TF-IDF слов на отзыв
  ungroup()

# Простые числовые признаки
model_data <- df %>%
  mutate(row_id = row_number()) %>%
  filter(sentiment != "Neutral") %>%
  left_join(
    review_afinn %>% rename(row_id_afinn = row_id),
    by = c("row_id" = "row_id_afinn", "sentiment")
  ) %>%
  mutate(
    afinn_score    = replace_na(afinn_score, 0),
    sentiment_bin  = ifelse(sentiment == "Positive", 1, 0),
    has_exclamation = as.integer(has_exclamation)
  ) %>%
  select(sentiment_bin, word_count, review_length, has_exclamation,
         afinn_score, helpful_yes, total_vote, day_diff) %>%
  drop_na()

cat("Датасет для моделирования:", nrow(model_data), "строк\n")
cat("Доля Positive:", mean(model_data$sentiment_bin), "\n\n")

# Разбивка Train/Test (80/20)
set.seed(42)
train_idx <- createDataPartition(model_data$sentiment_bin, p = 0.8, list = FALSE)
train_data <- model_data[train_idx, ]
test_data  <- model_data[-train_idx, ]

train_data$sentiment_bin <- factor(train_data$sentiment_bin, 
                                   labels = c("Negative", "Positive"))
test_data$sentiment_bin  <- factor(test_data$sentiment_bin,
                                   labels = c("Negative", "Positive"))

ctrl <- trainControl(method = "cv", number = 5, 
                     classProbs = TRUE, summaryFunction = twoClassSummary)

# Logistic Regression
cat("Обучение Logistic Regression...\n")
set.seed(42)
model_lr <- train(sentiment_bin ~ ., data = train_data,
                  method = "glm", family = "binomial",
                  trControl = ctrl, metric = "ROC")

pred_lr <- predict(model_lr, test_data)
cm_lr   <- confusionMatrix(pred_lr, test_data$sentiment_bin, positive = "Positive")
cat("\n Logistic Regression \n")
print(cm_lr)

# Naive Bayes
cat("\nОбучение Naive Bayes...\n")
set.seed(42)
model_nb <- train(sentiment_bin ~ ., data = train_data,
                  method = "naive_bayes",
                  trControl = ctrl, metric = "ROC")

pred_nb <- predict(model_nb, test_data)
cm_nb   <- confusionMatrix(pred_nb, test_data$sentiment_bin, positive = "Positive")
cat("\n Naive Bayes \n")
print(cm_nb)

# Random Forest 
cat("\nОбучение Random Forest...\n")
set.seed(42)
model_rf <- train(sentiment_bin ~ ., data = train_data,
                  method = "rf", ntree = 100,
                  trControl = ctrl, metric = "ROC")

pred_rf <- predict(model_rf, test_data)
cm_rf   <- confusionMatrix(pred_rf, test_data$sentiment_bin, positive = "Positive")
cat("\n Random Forest \n")
print(cm_rf)

# Важность признаков Random Forest
p9 <- ggplot(
  varImp(model_rf)$importance %>%
    rownames_to_column("feature") %>%
    arrange(desc(Overall)) %>%
    head(8),
  aes(x = reorder(feature, Overall), y = Overall)
) +
  geom_col(fill = "#3498db") +
  coord_flip() +
  labs(title = "Рис. 9. Важность признаков Random Forest",
       x = NULL, y = "Важность (Mean Decrease Gini)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(p9)
ggsave("plot9_feature_importance.png", p9, width = 7, height = 5, dpi = 150)

# Сравнение моделей
results_df <- data.frame(
  Model     = c("Logistic Regression", "Naive Bayes", "Random Forest"),
  Accuracy  = c(cm_lr$overall["Accuracy"],
                cm_nb$overall["Accuracy"],
                cm_rf$overall["Accuracy"]),
  Precision = c(cm_lr$byClass["Precision"],
                cm_nb$byClass["Precision"],
                cm_rf$byClass["Precision"]),
  Recall    = c(cm_lr$byClass["Recall"],
                cm_nb$byClass["Recall"],
                cm_rf$byClass["Recall"]),
  F1        = c(cm_lr$byClass["F1"],
                cm_nb$byClass["F1"],
                cm_rf$byClass["F1"])
)

cat("\n СРАВНЕНИЕ МОДЕЛЕЙ \n")
print(results_df %>% mutate(across(where(is.numeric), ~round(., 4))))

p10 <- results_df %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1),
               names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = round(Value, 3)), position = position_dodge(0.9),
            vjust = -0.4, size = 3) +
  scale_fill_brewer(palette = "Set2") +
  ylim(0, 1.05) +
  labs(title = "Рис. 10. Сравнение метрик моделей классификации",
       x = NULL, y = "Значение метрики", fill = "Метрика") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 10, hjust = 1))

print(p10)
ggsave("plot10_model_comparison.png", p10, width = 9, height = 6, dpi = 150)

# ИТОГОВЫЕ ВЫВОДЫ

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("ИТОГОВЫЕ ВЫВОДЫ\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

cat("1. ДАННЫЕ: Датасет содержит", nrow(df), "отзывов Amazon.\n")
cat("   Распределение тональности (%):\n")
print(round(prop.table(table(df$sentiment)) * 100, 1))

cat("\n2. EDA:\n")
cat("   - Датасет сильно несбалансирован: ~80% Positive-отзывов.\n")
cat("   - Negative-отзывы в среднем длиннее Positive.\n")
cat("   - AFINN-скоры чётко разделяют Positive и Negative группы.\n")

cat("\n3. ГИПОТЕЗЫ:\n")
cat("   - Длина отзыва значимо различается по группам (ANOVA, p <", 
    round(p_anova, 4), ")\n")
cat("   - AFINN-скор значимо выше у Positive-отзывов (U-тест, p <", 
    round(wilcox_result$p.value, 6), ")\n")

cat("\n4. ЛУЧШАЯ МОДЕЛЬ:", 
    results_df$Model[which.max(results_df$F1)],
    "— F1 =", round(max(results_df$F1), 4), "\n")

cat("\n5. Все графики сохранены в рабочую директорию (plot1_*.png ... plot10_*.png)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

