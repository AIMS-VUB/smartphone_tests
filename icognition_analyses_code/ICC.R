# Source: https://www.datanovia.com/en/lessons/intraclass-correlation-coefficient-in-r/

library(irr)

df_symbol_test <- read.csv('[SPECIFY_PATH_TO_OUTPUT_DIRECTORY]/icc_df_for_r_symbol_test_n_correct_v1_0_0_reject_True.csv')
df_dot_test <- read.csv('[SPECIFY_PATH_TO_OUTPUT_DIRECTORY]/icc_df_for_r_dot_test_n_correct_total_v1_0_0_reject_True.csv')
df_vbds <- read.csv('[SPECIFY_PATH_TO_OUTPUT_DIRECTORY]/icc_df_for_r_digit_span_vis_span_level_score_v1_0_0_reject_True.csv')

print('Symbol Test')
print('-----------')
icc(df_symbol_test, model="twoway", type="agreement", unit="single")
print('Dot Test')
print('-----------')
icc(df_dot_test, model="twoway", type="agreement", unit="single")
print('vBDS')
print('-----------')
icc(df_vbds, model="twoway", type="agreement", unit="single")

print('')
print('')
print('COGNITIVELY PRESERVED:')
print('')
print('')

df_symbol_test_cp <- read.csv('[SPECIFY_PATH_TO_OUTPUT_DIRECTORY]/icc_df_for_r_symbol_test_n_correct_v1_0_0_reject_True_cogpreserved.csv')
df_dot_test_cp <- read.csv('[SPECIFY_PATH_TO_OUTPUT_DIRECTORY]/icc_df_for_r_dot_test_n_correct_total_v1_0_0_reject_True_cogpreserved.csv')
df_vbds_cp <- read.csv('[SPECIFY_PATH_TO_OUTPUT_DIRECTORY]/icc_df_for_r_digit_span_vis_span_level_score_v1_0_0_reject_True_cogpreserved.csv')

print('Symbol Test')
print('-----------')
icc(df_symbol_test_cp, model="twoway", type="agreement", unit="single")
print('Dot Test')
print('-----------')
icc(df_dot_test_cp, model="twoway", type="agreement", unit="single")
print('vBDS')
print('-----------')
icc(df_vbds_cp, model="twoway", type="agreement", unit="single")
