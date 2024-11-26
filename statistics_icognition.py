"""
icognition statistics
"""

import os
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def calculate_correlation(a, b, type_corr):
    """ Calculate correlation between a and b depending on normality test (Shapiro-Wilk)

    :param a: list 1
    :param b: list 2
    :return: r, p, correlation test
    """

    if type_corr == 'Pearson':
        r_ab, p_ab = stats.pearsonr(a,b)
    elif type_corr == 'Spearman':
        r_ab, p_ab = stats.spearmanr(a,b)
    else:
        raise ValueError(f'Correlation type not understood: {type_corr}')
    return r_ab, p_ab


def z_normalise(df_hc_overall, df_to_normalise, target_colname, covariates_colnames, lower_is_better):
    """
    This function performs z-normalisation
    """

    # Reject lines with missing value on any of the variables
    df_hc = df_hc_overall.copy()
    df_hc = df_hc.dropna(subset=covariates_colnames + [target_colname], how='any')

    # Get tools for z-normalisation from
    model, diff_mean_hc, diff_std_hc, txt = get_healthy_model(df_hc, target_colname, covariates_colnames)

    # Loop over all rows in the data frame
    z_list = []
    for i, row in df_to_normalise.iterrows():
        row_df = row.to_frame().T
        # Split data
        y = row_df[target_colname].iloc[0]
        X = row_df[covariates_colnames]
        X = sm.add_constant(X, has_constant='add')  # https://github.com/statsmodels/statsmodels/issues/7057

        y_pred = model.predict(X)
        diff = y - y_pred.iloc[0]
        z = diff_norm = (diff - diff_mean_hc)/diff_std_hc

        if lower_is_better:
            z = -z
        z_list.append(z)

    # Save z-scores to dataframe
    df_to_normalise[f'z_{target_colname}'] = z_list

    return df_to_normalise, txt


def get_healthy_model(df_hc, target_colname, covariates_colnames):
    # Print information about the distribution of all columns
    print(df_hc[covariates_colnames + [target_colname]].describe())

    # Split data
    y = df_hc[target_colname]
    X = df_hc[covariates_colnames]
    X = sm.add_constant(X)

    # Fit linear regression
    model = sm.OLS(y, X)
    results = model.fit()

    # Predict target
    y_pred = results.predict(X)
    diff = y - y_pred
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)

    # Get text
    txt = f'\n\nLinear Regression Summary:\n\n {results.summary()}\n\n' \
          f'HC Diff Mean (SD): {diff_mean} ({diff_std})\n\n'

    return results, diff_mean, diff_std, txt


def test_impaired_or_not(x):
    """
    Define impairment based on Sumowski et al. 2018:
    https://doi.org/10.1212/WNL.0000000000004977
    """
    return x <= -1.5


def concurrent_validity(df_hc_overall, df_ms_overall, icognition_scores, paper_pencil_scores, txt, label_dict, output_dir_path,
                        reject_v1_0_0_for_dot_test, manual_filename_extension=''):
    """
    Calculate concurrent validity
    """

    if reject_v1_0_0_for_dot_test:
        filename_extension = '_v1.0.0_rejected'
    else:
        filename_extension = ''

    txt += '\n\n' \
           '###################\n' \
           'Concurrent Validity\n' \
           '###################\n\n'

    # Create figure
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    for score_icog, score_paper_pencil, row_nr in zip(icognition_scores, paper_pencil_scores, [0,1,2]):
        txt += f'\n\n{score_icog} versus {score_paper_pencil}\n\n'

        # Reject lines with missing value on one or both of the variables
        df_hc = df_hc_overall.copy()
        df_ms = df_ms_overall.copy()
        df_hc = df_hc.dropna(subset=[score_icog, score_paper_pencil], how='any')
        df_ms = df_ms.dropna(subset=[score_icog, score_paper_pencil], how='any')
        # Reject version number 1.0.0 of icognition for dot_test if indicated
        if reject_v1_0_0_for_dot_test:
            if score_icog == 'dot_test_n_correct_total':
                df_ms = df_ms[df_ms['version'] != '1.0.0']

        # Get sample sizes
        n_hc = df_hc.shape[0]
        n_ms = df_ms.shape[0]

        # Calculate correlation
        # Always Pearson for concurrent validity
        type_corr = 'Pearson'
        r_hc, p_hc = calculate_correlation(df_hc[score_icog], df_hc[score_paper_pencil], type_corr)
        r_ms, p_ms = calculate_correlation(df_ms[score_icog], df_ms[score_paper_pencil], type_corr)

        # Create visualisation
        sns.regplot(x=df_hc[score_icog], y=df_hc[score_paper_pencil], color='#51789D', fit_reg=True,
                    ax=axes[row_nr])
        sns.regplot(x=df_ms[score_icog], y=df_ms[score_paper_pencil], color='#DD8861', fit_reg=True,
                    ax=axes[row_nr])
        patch_hc = mpatches.Patch(color='#51789D', label='HC')
        patch_ms = mpatches.Patch(color='#DD8861', label='MS')

        axes[row_nr].legend(handles=[patch_hc, patch_ms], handleheight=0.2, handlelength=1)
        axes[row_nr].set_xlabel(label_dict.get(score_icog), fontsize=12)
        axes[row_nr].set_ylabel(label_dict.get(score_paper_pencil), fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add text
        txt += f'HC: r = {r_hc:.2f}, p = {p_hc:.3g}, n = {n_hc} ({type_corr})\n'
        txt += f'MS: r = {r_ms:.2f}, p = {p_ms:.3g}, n = {n_ms} ({type_corr})\n'

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_path, f'concurrent_validity{filename_extension}{manual_filename_extension}.png'), dpi=300)
    plt.close()

    return txt



def ms_versus_hc(df_hc_overall, df_ms_overall, scores, txt, label_dict, output_dir_path, reject_v1_0_0_for_dot_test,
                 z_normalised, manual_filename_extension=''):
    """ Calculate how test behaves in MS vs HC

    :param df_hc_overall: pandas dataframe, healthy controls
    :param df_ms_overall: pandas dataframe, people with MS
    :param scores: list, variables to plot
    :return:
    """

    if reject_v1_0_0_for_dot_test:
        filename_extension = '_v1.0.0_rejected'
    else:
        filename_extension = ''

    txt += '\n\n' \
           '############\n' \
           'MS versus HC\n' \
           '############\n\n'
    # Create figure
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    for score, row_nr in zip(scores, [0, 1, 2]):
        txt += f'\n\n{score}: MS versus HC\n\n'

        # Reject lines with missing value on one or both of the variables
        df_hc = df_hc_overall.copy()
        df_ms = df_ms_overall.copy()
        df_hc = df_hc[df_hc[score].notna()]
        df_ms = df_ms[df_ms[score].notna()]

        # Reject version number 1.0.0 of icognition for dot_test if indicated
        if reject_v1_0_0_for_dot_test:
            if score == 'dot_test_n_correct_total':
                df_ms = df_ms[df_ms['version'] != '1.0.0']

        n_hc = df_hc.shape[0]
        n_ms = df_ms.shape[0]

        # Compare distributions
        stat, p = stats.mannwhitneyu(df_hc[score], df_ms[score])

        # Create visualisation
        sns.kdeplot(x=df_hc[score], color='#51789D', ax=axes[row_nr])
        sns.kdeplot(x=df_ms[score], color='#DD8861', ax=axes[row_nr])
        patch_hc = mpatches.Patch(color='#51789D', label='HC')
        patch_ms = mpatches.Patch(color='#DD8861', label='MS')
        axes[row_nr].legend(handles=[patch_hc, patch_ms], handleheight=0.2, handlelength=1)
        axes[row_nr].set_xlabel(label_dict.get(score), fontsize=12)
        axes[row_nr].set_ylabel('Density', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add text
        txt += f'{score}: U = {stat:.2f}, p = {p:.3g}, n_hc = {n_hc}, n_ms = {n_ms}, (Mann-Whitney U)\n'

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_path, f'ms_versus_hc{filename_extension}_z={z_normalised}{manual_filename_extension}.png'), dpi=300)
    plt.close()

    return txt


def ecological_validity(df_overall, var_list_1, other_var, txt, label_dict, output_dir_path, reject_v1_0_0_for_dot_test,
                        type_corr, manual_filename_extension=''):
    """
    Calculate ecological validity
    """

    if reject_v1_0_0_for_dot_test:
        filename_extension = '_v1.0.0_rejected'
    else:
        filename_extension = ''

    txt += '\n\n' \
           '###################\n' \
           'Ecological Validity\n' \
           '###################\n\n'
    # Create figure
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    var_list_2 = [other_var]*len(var_list_1)
    for var_1, var_2, row_nr in zip(var_list_1, var_list_2, [0, 1, 2]):
        txt += f'\n\n{var_1} versus {var_2}\n\n'
        # Reject lines with missing value on one or both of the variables
        df = df_overall.copy()
        df = df.dropna(subset=[var_1, var_2], how='any')

        # Reject version number 1.0.0 of icognition for dot_test if indicated
        if reject_v1_0_0_for_dot_test:
            if var_1 == 'dot_test_n_correct_total':
                df = df[df['version'] != '1.0.0']

        n = df.shape[0]

        # Calculate correlation
        r, p = calculate_correlation(df[var_1], df[var_2], type_corr)

        # Create visualisation
        sns.regplot(x=df[var_1], y=df[var_2], color='#DD8861',
                    fit_reg=True, ax=axes[row_nr])
        patch_ms = mpatches.Patch(color='#DD8861', label='MS')
        axes[row_nr].legend(handles=[patch_ms], handleheight=0.2, handlelength=1)
        axes[row_nr].set_xlabel(label_dict.get(var_1), fontsize=12)
        axes[row_nr].set_ylabel(label_dict.get(var_2), fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add text
        txt += f'r = {r:.2f}, p = {p:.3g}, n = {n} ({type_corr})\n'

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_path, f'ecological_validity_{other_var}{filename_extension}{manual_filename_extension}.png'), dpi=300)
    plt.close()

    return txt


def test_retest_reliability(df_overall, icognition_scores, txt, label_dict, output_dir_path, reject_v1_0_0_for_dot_test, manual_filename_extension=''):
    """ Calculate test-retest reliability

    :param df_overall: pandas dataframe, people with MS
    :param icognition_scores: list, column names of icognition scores
    :return:
    """

    if reject_v1_0_0_for_dot_test:
        filename_extension = '_v1.0.0_rejected'
    else:
        filename_extension = ''

    txt += '\n\n' \
           '#######################\n' \
           'Test-retest Reliability\n' \
           '#######################\n\n'
    # Create figure
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    for var_baseline, row_nr in zip(icognition_scores, [0, 1, 2]):

        var_retest = f'{var_baseline}_retest'
        txt += f'\n\n{var_baseline} versus {var_retest}\n\n'

        # Reject lines with missing value on one or both of the variables
        df = df_overall.copy()
        df = df.dropna(subset=[var_baseline, var_retest], how='any')

        # Reject version number 1.0.0 of icognition for dot_test if indicated
        if reject_v1_0_0_for_dot_test:
            if var_baseline == 'dot_test_n_correct_total':
                df = df[df['version'] != '1.0.0']

        n = df.shape[0]

        # Calculate correlation
        type_corr = 'Pearson'
        r, p = calculate_correlation(df[var_baseline], df[var_retest], type_corr)

        # Create visualisation
        sns.regplot(x=df[var_baseline], y=df[var_retest], color='#51789D', fit_reg=True,
                    ax=axes[row_nr])
        patch_hc = mpatches.Patch(color='#51789D', label='HC')
        axes[row_nr].legend(handles=[patch_hc], handleheight=0.2, handlelength=1)
        axes[row_nr].set_xlabel(label_dict.get(var_baseline), fontsize=12)
        axes[row_nr].set_ylabel(label_dict.get(var_baseline) + ' retest', fontsize=12)
        axes[row_nr] = adapt_ax(axes[row_nr])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add text
        txt += f'r = {r:.2f}, p = {p:.3g}, n = {n} ({type_corr})\n'

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_path, f'test-retest_reliability{filename_extension}{manual_filename_extension}.png'), dpi=300)
    plt.close()

    return txt


def adapt_ax(ax):
    """
    Function created from following source: https://stackoverflow.com/questions/25497402/
    """
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    # now plot both limits against eachother
    ax.plot(lims, lims, '--', alpha=0.2, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    return ax


def get_var_info(df, var, description_method):

    if description_method == 'describe':
        txt = f'\n\nVariable: {var}\n' \
              f'{df[var].describe()}\n'
    elif description_method == 'value_counts':
        txt = f'\n\nVariable: {var}\n' \
              f'{df[var].value_counts()}\n'
    else:
        raise ValueError('Description method not understood')

    return txt


def compare_variable_hc_ms(df_hc_overall, df_ms_overall, var, test_type):

    df_hc = df_hc_overall.copy()
    df_ms = df_ms_overall.copy()
    df_hc = df_hc.dropna(subset = [var])
    df_ms = df_ms.dropna(subset = [var])
    n_hc = df_hc.shape[0]
    n_ms = df_ms.shape[0]
    txt = f'Number of HC: {n_hc}\n' \
          f'Number of people with MS: {n_ms}\n\n'

    if test_type == 'chi_square':
        chi2, p, contingency_table = calculate_chi_square(df_hc, df_ms, var)
        txt += f'Chi2 = {chi2}, p = {p}\n' \
               f'{contingency_table}'
    elif test_type == 'Independent t-test two-sided':
        t, p = stats.ttest_ind(df_hc[var], df_ms[var])
        txt += f'Independent T-test two-sided: t = {t}, p = {p}'
    elif test_type == 'Mann-Whitney U two-sided':
        U, p = stats.mannwhitneyu(df_hc[var], df_ms[var])
        txt += f'Mann-Whitney U two-sided: U = {U}, p = {p}'
    else:
        raise ValueError('Var type not understood')

    return txt


def calculate_chi_square(df1, df2, variable):
    # CAVE: method will fail if not both labels present in df1
    # Method should not be affected by NaN
    lab_1, lab_2 = df1[variable].value_counts().index
    contingency_table = pd.DataFrame({lab_1: [df1[variable].value_counts().loc[lab_1],
                                              df2[variable].value_counts().loc[lab_1]],
                                      lab_2: [df1[variable].value_counts().loc[lab_2],
                                              df2[variable].value_counts().loc[lab_2]]}, index = ['df_1', 'df_2'])
    chi2, p, _, _ = stats.chi2_contingency(contingency_table.T)
    return chi2, p, contingency_table


def calculate_icc(df, test_column, retest_column):
    # Create empty dataframe
    icc_df = pd.DataFrame()

    # Fill dataframe
    exam = 0
    for i, row in df.iterrows():
        test_value = row[test_column]
        retest_value = row[retest_column]
        if not np.isnan(test_value) and not np.isnan(retest_value):
            icc_df = pd.concat([icc_df, pd.DataFrame({'exam': [exam], 'judge': ['baseline'], 'rating': [test_value]})])
            icc_df = pd.concat(
                [icc_df, pd.DataFrame({'exam': [exam], 'judge': ['follow-up'], 'rating': [retest_value]})])
            exam += 1

    # Calculate ICC
    icc = pg.intraclass_corr(data=icc_df, targets='exam', raters='judge', ratings='rating')
    print('Number of subjects: ', icc_df.shape[0] / 2)
    print(icc.set_index('Type'))
    print(icc_df)
    return icc_df, icc


def icc_for_r_analysis(df, test_column, retest_column):
    return df[[test_column, retest_column]].dropna()


if __name__ == "__main__":

    output_path = '[SPECIFY PATH TO OUTPUT DIRECTORY]'

    # Read dataframe
    df = pd.read_excel('[SPECIFY PATH TO INPUT EXCEL FILE]')

    # Split data in HC and MS
    df_hc = df[df['group'] == 'hc']
    df_ms = df[df['group'] == 'ms']

    # Remove excluded subjects
    df_hc = df_hc[df_hc['rejection'] == 'no']
    df_ms = df_ms[df_ms['rejection'] == 'no']

    # Add sex_num variable for regression-based normalisation
    df_hc['sex_num'] = df_hc['sex'].apply(lambda x: {'m': 1, 'v': 2}.get(x))
    df_ms['sex_num'] = df_ms['sex'].apply(lambda x: {'m': 1, 'v': 2}.get(x))

    # Initialise txt for txt file
    txt = f'Number of people with MS: {df_ms.shape[0]}\n'\
          f'Number of healthy controls: {df_hc.shape[0]}\n\n'

    # Sample characteristics
    for var, description_method, test_type in zip(
            ['sex_num', 'age', 'education_n_years', 'sdmt', 'spart_tot', 'digit_span_vis_span_level_score',
             'bdi_total', 'fsmc_total', 'symbol_test_n_correct', 'dot_test_n_correct_total', 'digit_span_aud_span_level_score'],
            ['value_counts'] + ['describe'] * 10,
            ['chi_square'] + ['Mann-Whitney U two-sided'] * 10):

        txt += f'\n\n' \
               f'##############\n' \
               f'Variable: {var}\n' \
               f'##############\n\n'
        df_ms_for_var_comparison = df_ms.copy()
        if var == 'dot_test_n_correct_total':
            df_ms_for_var_comparison = df_ms_for_var_comparison[df_ms_for_var_comparison['version'] != '1.0.0']
        txt += '\n--> Multiple Sclerosis\n\n'
        txt += get_var_info(df_ms_for_var_comparison, var, description_method)
        txt += '--> Healthy Controls\n\n'
        txt += get_var_info(df_hc, var, description_method)
        txt += '\n--> Comparison\n\n'
        txt += compare_variable_hc_ms(df_hc, df_ms_for_var_comparison, var, test_type)

    # HC-specific information
    txt += '\n\n**************************' \
           'HC-specific' \
           '**************************\n\n'
    txt += get_var_info(df_hc, 'test-retest_time_difference_days', 'describe')

    # MS-specific information
    txt += '\n\n**************************' \
           'MS-specific' \
           '**************************\n\n'
    txt += get_var_info(df_ms, 'disease_duration', 'describe')
    txt += get_var_info(df_ms, 'type_ms', 'value_counts')
    txt += get_var_info(df_ms, 'edss', 'describe')

    # Initialise variables
    icognition_scores = ['symbol_test_n_correct', 'dot_test_n_correct_total', 'digit_span_vis_span_level_score']
    paper_pencil_scores = ['sdmt', 'spart_tot', 'digit_span_aud_span_level_score']
    icognition_labels = ['Symbol Test', 'Dot Test', 'vBDS']
    paper_pencil_labels = ['SDMT', 'Spart 10/36', 'aBDS']
    label_dict = dict(zip(icognition_scores + paper_pencil_scores + ['edss', 'bdi_total', 'fsmc_total', 'disease_duration', 'education_n_years', 'age'],
                          icognition_labels + paper_pencil_labels + ['EDSS', 'BDI', 'FSMC', 'Disease Duration', 'Education Level', 'Age']))

    # Regression-based normalisation
    txt += '\n\n*******************\n' \
           'Regression analysis\n' \
           '*******************\n\n'
    for score in icognition_scores + paper_pencil_scores:
        df_ms, txt_ms = z_normalise(df_hc, df_ms, score, ['sex_num', 'education_n_years', 'age'], False)
        df_hc, txt_hc = z_normalise(df_hc, df_hc, score, ['sex_num', 'education_n_years', 'age'], False)
        txt += txt_ms + txt_hc
        df_ms[f'{score}_impaired'] = df_ms[f'z_{score}'].apply(test_impaired_or_not)
        df_hc[f'{score}_impaired'] = df_hc[f'z_{score}'].apply(test_impaired_or_not)
        txt += f'N impaired MS ({score}): {df_ms[f"{score}_impaired"].sum()}\n'
        txt += f'N impaired HC ({score}): {df_hc[f"{score}_impaired"].sum()}\n'

    for reject_v1_0_0 in [True, False]:
        txt += f'\n\n--> REJECT V1.0.0 = {reject_v1_0_0}\n\n'
        # Concurrent validity
        txt = concurrent_validity(df_hc, df_ms, icognition_scores, paper_pencil_scores, txt, label_dict, output_path, reject_v1_0_0_for_dot_test=reject_v1_0_0)

        # Ecological validity
        txt = ecological_validity(df_ms, icognition_scores, 'edss', txt, label_dict, output_path, type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0)
        txt = ecological_validity(df_ms, icognition_scores, 'bdi_total', txt, label_dict, output_path, type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0)
        txt = ecological_validity(df_ms, icognition_scores, 'fsmc_total', txt, label_dict, output_path, type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0)
        txt = ecological_validity(df_ms, icognition_scores, 'disease_duration', txt, label_dict, output_path, type_corr='Pearson', reject_v1_0_0_for_dot_test=reject_v1_0_0)
        txt = ecological_validity(df_ms, icognition_scores, 'education_n_years', txt, label_dict, output_path, type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0)
        txt = ecological_validity(df_ms, icognition_scores, 'age', txt, label_dict, output_path, type_corr='Pearson', reject_v1_0_0_for_dot_test=reject_v1_0_0)

        # test-retest reliability
        txt = test_retest_reliability(df_hc, icognition_scores, txt, label_dict, output_path, reject_v1_0_0_for_dot_test=reject_v1_0_0)
        for icognition_score in icognition_scores:
            _, icc = calculate_icc(df_hc, icognition_score, f'{icognition_score}_retest')
            txt += f'\n>> ICC ({icognition_score}): {icc}\n\n'
            df_test_retest = df_hc[df_hc[[icognition_score, f'{icognition_score}_retest']].notna().sum(axis=1)==2]
            W_test_retest, p_test_retest = stats.wilcoxon(df_test_retest[icognition_score], df_test_retest[f'{icognition_score}_retest'])
            txt += f'>> Wilcoxon baseline - retest: W = {W_test_retest}, p = {p_test_retest}\n'

            # For R analysis of ICC
            icc_df_for_r = icc_for_r_analysis(df_hc, icognition_score, f'{icognition_score}_retest')
            icc_df_for_r.to_csv(os.path.join(output_path, f'icc_df_for_r_{icognition_score}_v1_0_0_reject_{reject_v1_0_0}.csv'), index=False)

        # MS versus HC
        # - icognition test scores
        txt = ms_versus_hc(df_hc, df_ms, icognition_scores, txt, label_dict, output_path, reject_v1_0_0_for_dot_test=reject_v1_0_0, z_normalised=False)
        # - regression-based norms icognition test scores
        label_dict_regression_based_norms = {f'z_{key}': f'{value} (Z-score)' for key, value in dict(zip(icognition_scores, icognition_labels)).items()}
        txt = ms_versus_hc(df_hc, df_ms, label_dict_regression_based_norms.keys(), txt, label_dict_regression_based_norms, output_path, reject_v1_0_0_for_dot_test=reject_v1_0_0, z_normalised=True)
        # - paper-pencil test scores
        txt = ms_versus_hc(df_hc, df_ms, paper_pencil_scores, txt, label_dict, output_path, reject_v1_0_0_for_dot_test=reject_v1_0_0, z_normalised=False, manual_filename_extension='_paper-pencil')

        # Cognitive impairment analysis
        if reject_v1_0_0:
            impairment_columns = [f'{test}_impaired' for test in paper_pencil_scores]
            df_hc_non_impaired = df_hc[df_hc[impairment_columns].sum(axis=1) == 0]
            df_ms_non_impaired = df_ms[df_ms[impairment_columns].sum(axis=1) == 0]

            txt += '\n\n%%%%%%%%%%%%%%%%%%%%%%%%\nWithout cognitively impaired subjects\n%%%%%%%%%%%%%%%%%%%%%%%%\n\n'
            # Concurrent
            txt = concurrent_validity(df_hc_non_impaired, df_ms_non_impaired, icognition_scores, paper_pencil_scores, txt, label_dict,
                                      output_path, reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')

            # Ecological
            txt = ecological_validity(df_ms_non_impaired, icognition_scores, 'edss', txt, label_dict, output_path,
                                      type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')
            txt = ecological_validity(df_ms_non_impaired, icognition_scores, 'bdi_total', txt, label_dict, output_path,
                                      type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')
            txt = ecological_validity(df_ms_non_impaired, icognition_scores, 'fsmc_total', txt, label_dict, output_path,
                                      type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')
            txt = ecological_validity(df_ms_non_impaired, icognition_scores, 'disease_duration', txt, label_dict, output_path,
                                      type_corr='Pearson', reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')
            txt = ecological_validity(df_ms_non_impaired, icognition_scores, 'education_n_years', txt, label_dict, output_path,
                                      type_corr='Spearman', reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')
            txt = ecological_validity(df_ms_non_impaired, icognition_scores, 'age', txt, label_dict, output_path,
                                      type_corr='Pearson', reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')

            # test-retest reliability
            txt = test_retest_reliability(df_hc_non_impaired, icognition_scores, txt, label_dict, output_path,
                                          reject_v1_0_0_for_dot_test=reject_v1_0_0, manual_filename_extension='_cogpreserved')
            for icognition_score in icognition_scores:
                _, icc = calculate_icc(df_hc_non_impaired, icognition_score, f'{icognition_score}_retest')
                txt += f'\n>> ICC ({icognition_score}): {icc}\n\n'
                df_test_retest_non_impaired = df_hc_non_impaired[df_hc_non_impaired[[icognition_score, f'{icognition_score}_retest']].notna().sum(axis=1) == 2]
                W_test_retest_non_imp, p_test_retest_non_imp = stats.wilcoxon(df_test_retest_non_impaired[icognition_score], df_test_retest_non_impaired[f'{icognition_score}_retest'])
                txt += f'>> Wilcoxon baseline - retest: W = {W_test_retest_non_imp}, p = {p_test_retest_non_imp}\n'

                # For R analysis of ICC
                icc_df_for_r = icc_for_r_analysis(df_hc_non_impaired, icognition_score, f'{icognition_score}_retest')
                icc_df_for_r.to_csv(
                    os.path.join(output_path, f'icc_df_for_r_{icognition_score}_v1_0_0_reject_{reject_v1_0_0}_cogpreserved.csv'),
                    index=False)

            # MS versus HC
            # - icognition test scores
            txt = ms_versus_hc(df_hc_non_impaired, df_ms_non_impaired, icognition_scores, txt, label_dict, output_path,
                               reject_v1_0_0_for_dot_test=reject_v1_0_0, z_normalised=False, manual_filename_extension='_cogpreserved')

    with open(os.path.join(output_path, 'output.txt'), 'w') as fp:
        fp.write(txt)
