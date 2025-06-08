# 3PL IRT Parameter Estimation with mirt (Processamento em Paralelo e Cálculo de Nota)
# ================================================================
# Este script estima parâmetros de um modelo 3PL usando {mirt},
# acelera o processamento com paralelização e gera notas de 0 a 1000.

# 1. Instalação e carregamento de pacotes ------------------------------
if (!requireNamespace("mirt", quietly = TRUE)) install.packages("mirt")
if (!requireNamespace("parallel", quietly = TRUE)) install.packages("parallel")

library(mirt)
library(parallel)

# 2. Configuração de computação paralela --------------------------------
# Usa todos os núcleos menos um para o sistema
nCores <- detectCores() - 1
cl <- makeCluster(nCores)
mirtCluster(cl)

# 3. Carregamento da base de respostas ----------------------------------
# CSV com cada linha = aluno, cada coluna = questão (0/1)
file_path <- "tri/enem_tri/respostas_simuladas.csv"
resp_df <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
resp <- as.matrix(resp_df)
mode(resp) <- "numeric"
rownames(resp) <- paste0("Aluno", seq_len(nrow(resp)))

# 4. Ajuste do modelo 3PL -----------------------------------------------
model_3pl <- mirt(data = resp,
                  model = 1,
                  itemtype = "3PL",
                  SE = TRUE)

# 5. Extração dos parâmetros -------------------------------------------
pars <- coef(model_3pl, IRTpars = TRUE, simplify = TRUE)$items
print(pars)

# 6. Função da fórmula 3PL ----------------------------------------------
prob_3pl <- function(theta, a, b, c, D = 1.7) {
  c + (1 - c) * (1 / (1 + exp(-D * a * (theta - b))))
}

# 7. Estimativa de habilidade (θ) por aluno -----------------------------
theta_est <- fscores(model_3pl, method = "EAP")
theta_values <- theta_est[, "F1"]

# 8. Cálculo de nota esperada e escala 0-1000 ----------------------------
n_items <- ncol(resp)
expected_raw <- sapply(theta_values, function(th) {
  sum(prob_3pl(th, pars[, "a"], pars[, "b"], pars[, "g"]))
})
score_0_1000 <- (expected_raw / n_items) * 1000

# 9. Resultados em data.frame -------------------------------------------
results <- data.frame(
  Aluno = rownames(resp),
  Theta = theta_values,
  ExpectedRaw = expected_raw,
  Score0to1000 = score_0_1000
)
# print(results)

# 10. Salvando resultados ----------------------------------------------
write.csv(results, "tri/enem_tri/notas_3PL.csv", row.names = FALSE)

# 11. Finalização do cluster paralelo ----------------------------------
stopCluster(cl)
mirtCluster(remove = TRUE)

# Fim do script.
