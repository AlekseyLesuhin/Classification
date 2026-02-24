import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def get_statistic(df: pd.DataFrame)-> pd.DataFrame:
    df_info = pd.DataFrame(df.notnull().sum(), columns = ['Заполненные значения'])
    df_info = df_info.join(pd.DataFrame(df.isnull().sum(), columns = ['Пропуски'])) . \
    join(pd.DataFrame(round(100*df.isnull().sum ()/ len(df),2), columns = ['Доля пропусков, %'])) . \
    join(pd.DataFrame(df.dtypes, columns=['Dtypes'])) . \
    join(pd.DataFrame(df.nunique(), columns=['Кол-во уникальных значений'])) #. \
    df_info['Мода'] = df.mode().iloc[0]
    df_info['Кол-во значений моды'] = df_info.apply(lambda row: (df[row.name]==row['Мода']).sum(), axis=1)
    df_info = df_info.join(df.describe().T[['mean','min','max']], how='left').replace(np.nan, '-')
    df_info.rename(columns={'mean': 'average'}, inplace=True)
    percentil_25 = []
    percentil_75 = []
    iqr=[]
    l_values=[]
    h_values=[]
    for col in df.columns:
        if col in df.select_dtypes(include='number').columns:
            quants = list(df[col].quantile([0.25, 0.75]))
            IQR = quants[1]-quants[0]
            lower_bound = quants[0]-1.5*IQR
            upper_bound = quants[1]+1.5*IQR
            if min(df[col])<lower_bound:
                low_values = f'ниже {lower_bound}'
            else:
                low_values = '-'
            if max(df[col])>upper_bound:
                high_values = f'выше {upper_bound}'
            else:
                high_values = '-'
            percentil_25.append(quants[0])
            percentil_75.append(quants[1])
            iqr.append(IQR)
            l_values.append(low_values)
            h_values.append(high_values)
        else:
            percentil_25.append('-')
            percentil_75.append('-')
            l_values.append('-')
            h_values.append('-')
            iqr.append('-')
    df_info['25-ый перцентиль'] = percentil_25
    df_info['75-ый перцентиль'] = percentil_75
    df_info['Межквартильное расстояние'] = iqr
    df_info['Межквартильное расстояние'] = iqr
    df_info['Выбросы (bottom)'] = l_values
    df_info['Выбросы (top)'] = h_values

    return df_info

def get_hist_plots(df: pd.DataFrame):
    cols = df.columns
    plots_per_fig = 9
    n_figs = math.ceil(len(cols) / plots_per_fig)
    
    with PdfPages('Histplots.pdf') as pdf:
        for i in range(n_figs):
            subset = cols[i*plots_per_fig:(i+1)*plots_per_fig]
    
            fig, axes = plt.subplots(
                nrows=3, ncols=3,
                figsize=(24, 18)
            )
    
            for ax, col in zip(axes.flatten(), subset):
                sns.histplot(df[col], ax=ax)
                ax.set_title(col, fontsize=18)
                ax.set_xlabel('value', fontsize=14)
                ax.set_ylabel("COUNTS", fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
    
            # удалить пустые оси
            for ax in axes.flatten()[len(subset):]:
                ax.remove()
    
            plt.tight_layout()
            #pdf.savefig(fig, dpi=300)
            plt.show()
            plt.close(fig)

def get_box_plots(df: pd.DataFrame):
    cols = df.select_dtypes(include='number').columns
    plots_per_fig = 9
    n_figs = math.ceil(len(cols) / plots_per_fig)
    
    with PdfPages('Boxplots.pdf') as pdf:
        for i in range(n_figs):
            subset = cols[i*plots_per_fig:(i+1)*plots_per_fig]
    
            fig, axes = plt.subplots(
                nrows=3, ncols=3,
                figsize=(24, 18)
            )
    
            for ax, col in zip(axes.flatten(), subset):
                sns.boxplot(data = df[col], ax=ax)
                ax.set_title(col, fontsize=18)
                ax.set_xlabel(col, fontsize=14)
                ax.set_ylabel("Value", fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
    
            # удалить пустые оси
            for ax in axes.flatten()[len(subset):]:
                ax.remove()
    
            plt.tight_layout()
            #pdf.savefig(fig, dpi=300)
            plt.show()
            plt.close(fig)

def get_corr_matrix(df: pd.DataFrame)->pd.DataFrame:
    df_corr = df[df.select_dtypes(include='number').columns].corr()
    plt.figure(figsize=(40,30))
    sns.heatmap(df_corr, \
                annot=True, \
                cmap='coolwarm', \
                cbar=True, \
                fmt='.1g', \
                annot_kws={"size":15})
    plt.title('Матрица корреляции', fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.savefig('Матрица корреляции.jpg', dpi=300)
    plt.show()

def get_count_plot(df):
    cols = list(df.select_dtypes(exclude='number').columns)
    cols.remove('deposit')
    plots_per_fig = 9
    n_figs = math.ceil(len(cols) / plots_per_fig)
    
    with PdfPages('Boxplots.pdf') as pdf:
        for i in range(n_figs):
            subset = cols[i*plots_per_fig:(i+1)*plots_per_fig]
    
            fig, axes = plt.subplots(
                nrows=3, ncols=3,
                figsize=(24, 18)
            )
    
            for ax, col in zip(axes.flatten(), subset):
                sns.countplot(data = df, x = col, hue='deposit', ax=ax)
                ax.set_title(col, fontsize=18)
                ax.set_xlabel(col, fontsize=14)
                ax.set_ylabel("Count", fontsize=14)
                ax.tick_params(axis='x', rotation=45, labelsize=12)
    
            # удалить пустые оси
            for ax in axes.flatten()[len(subset):]:
                ax.remove()
    
            plt.tight_layout()
            plt.show()
            plt.close(fig)

def get_violin_plot(df):
    cols = list(df.select_dtypes(include='number').columns)
    plots_per_fig = 9
    n_figs = math.ceil(len(cols) / plots_per_fig)
    
    with PdfPages('Boxplots.pdf') as pdf:
        for i in range(n_figs):
            subset = cols[i*plots_per_fig:(i+1)*plots_per_fig]
    
            fig, axes = plt.subplots(
                nrows=3, ncols=3,
                figsize=(24, 18)
            )
    
            for ax, col in zip(axes.flatten(), subset):
                sns.violinplot(data=df, x='deposit', y=col, ax=ax)
                ax.set_title(col, fontsize=18)
                ax.set_xlabel('deposit', fontsize=14)
                ax.set_ylabel(col, fontsize=14)
                ax.tick_params(axis='x', labelsize=12)
    
            # удалить пустые оси
            for ax in axes.flatten()[len(subset):]:
                ax.remove()
    
            plt.tight_layout()
            plt.show()
            plt.close(fig)

def get_kde_plot(df):
    cols = list(df.select_dtypes(include='number').columns)
    plots_per_fig = 9
    n_figs = math.ceil(len(cols) / plots_per_fig)
    
    with PdfPages('Boxplots.pdf') as pdf:
        for i in range(n_figs):
            subset = cols[i*plots_per_fig:(i+1)*plots_per_fig]
    
            fig, axes = plt.subplots(
                nrows=3, ncols=3,
                figsize=(24, 18)
            )
    
            for ax, col in zip(axes.flatten(), subset):
                sns.kdeplot(data=df, x=col, hue='deposit', fill=True, ax=ax)
                ax.set_title(col, fontsize=18)
                ax.set_xlabel('deposit', fontsize=14)
                ax.set_ylabel(col, fontsize=14)
                ax.tick_params(axis='x', labelsize=12)
    
            # удалить пустые оси
            for ax in axes.flatten()[len(subset):]:
                ax.remove()
    
            plt.tight_layout()
            plt.show()
            plt.close(fig)
