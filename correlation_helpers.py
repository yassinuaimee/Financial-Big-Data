import os
import polars as pl
from wrangling_helpers import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from numpy import linalg as LA

def compute_C_minus_C0(lambdas,v,lambda_plus,removeMarketMode=True):
    N=len(lambdas)
    C_clean=np.zeros((N, N))
    
    order = np.argsort(lambdas)
    lambdas,v = lambdas[order],v[:,order]
    
    v_m=np.matrix(v)

    # note that the eivenvalues are sorted
    for i in range(1*removeMarketMode,N):                            
        if lambdas[i]>lambda_plus: 
            C_clean=C_clean+lambdas[i] * np.dot(v_m[:,i],v_m[:,i].T)  
    return C_clean   

def eigenvalue_clipping(lambdas,v,lambda_plus):
    N=len(lambdas)
    
    # _s stands for _structure below
    sum_lambdas_gt_lambda_plus=np.sum(lambdas[lambdas>lambda_plus])
    
    sel_bulk=lambdas<=lambda_plus                     # these eigenvalues come from the seemingly random bulk
    N_bulk=np.sum(sel_bulk)
    sum_lambda_bulk=np.sum(lambdas[sel_bulk])        
    delta=sum_lambda_bulk/N_bulk                      # delta is their average, so as to conserver the trace of C
    
    lambdas_clean=lambdas
    lambdas_clean[lambdas_clean<=lambda_plus]=delta
    
    
    C_clean=np.zeros((N, N))
    v_m=np.matrix(v)
    
    for i in range(N-1):
        C_clean=C_clean+lambdas_clean[i] * np.dot(v_m[i,].T,v_m[i,]) 
        
    np.fill_diagonal(C_clean,1)
            
    return C_clean  

def P0(lambdas,q):
    lambda_plus = (1+np.sqrt(q))**2
    lambda_minus = (1-np.sqrt(q))**2
    vals = 1/(q*2*np.pi*lambdas)*np.sqrt((lambda_plus-lambdas)*(lambdas-lambda_minus))
    return vals

def plot_corr_heatmap(corr,df,on_timestamp,ax):
        if on_timestamp:
            sns.heatmap(corr, ax=ax,
            mask=np.triu(corr),
            vmin=-1,
            vmax=1,
            center= 0)
        else:

            sns.heatmap(corr, ax=ax,
            xticklabels=df.columns[1:],
            yticklabels=df.columns[1:],
            mask=np.triu(corr),
            vmin=-1,
            vmax=1,
            center= 0)
        ax.set_title('Cleaned Correlations')

def plot_MP_dist(lambdas_emp,lambdas_mp,p0s,ax):
        sns.histplot(lambdas_emp, kde=False, stat="probability", bins=100, binrange=(0, 20),ax=ax)
        ax.plot(lambdas_mp,p0s,label="Marcenko-Pastur Distribution",color='red')
        ax.set_title('Marcenko-Pastur EV distribution vs. Observed Distribution ')
        ax.legend()

def ev_cleaning_plotting(lambdas_emp,clean_ev,lambda_plus,ax):
        ax.plot(np.sort(lambdas_emp)[::-1], label="Original Eigenvalues", marker='o')
        ax.plot(np.sort(clean_ev)[::-1], label="Cleaned Eigenvalues", marker='x')
        ax.axhline(lambda_plus, color='red', linestyle='--', label="Lambda Plus")
        ax.set_ylabel("Eigenvalue")
        ax.legend()
        ax.set_title("Original vs. Cleaned Eigenvalues")

def plot_MP_clean(clean_ev,lambdas_mp,p0s,ax):
        sns.histplot(clean_ev, kde=False, stat="probability", bins=100, binrange=(0, 20),ax=ax)
        ax.plot(lambdas_mp, p0s, label="Marcenko-Pastur Distribution", color='red')
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Probability")
        ax.set_title("Cleaned Eigenvalues vs. Marcenko-Pastur Distribution")
        ax.legend()

def ev_cleaning(returns,on_timestamp=False,ev_clipping=True,MarketMode=True):
        fig,axs=plt.subplots(2,2,figsize=(10,10))
        if on_timestamp:
                corr_mx=np.corrcoef(returns.drop('index'),rowvar=on_timestamp)
        else:
                corr_mx=np.corrcoef(returns,rowvar=on_timestamp)
   
        
        #plot original correlation heatmap.

        lambdas_empirical, V_emprirical=LA.eig(corr_mx)
        q=returns.shape[0]/returns.shape[1]
        lambdas=np.linspace((1.-np.sqrt(q))**2,(1.+np.sqrt(q))**2,200)
        P0s=P0(lambdas,q)

        #plot theoritical ev distribution again empirical distribution
        plot_MP_dist(lambdas_empirical,lambdas,P0s,axs[0][0])

        lambda_plus=(1+np.sqrt(q))**2
        if ev_clipping:   
            C_clipped=eigenvalue_clipping(lambdas_empirical,V_emprirical,lambda_plus)
        else:
            C_clipped=compute_C_minus_C0(lambdas_empirical,V_emprirical,lambda_plus,removeMarketMode=MarketMode) # it works

        plot_corr_heatmap(C_clipped,returns,on_timestamp,axs[0][1])

        #Eigenvalues after cleaning.
        eigenvalues_clean,v_clean = LA.eig(C_clipped)

        # Plot original and cleaned eigenvalues
        ev_cleaning_plotting(lambdas_empirical,eigenvalues_clean,lambda_plus=lambda_plus,ax=axs[1][0])

        # Plot histogram of cleaned eigenvalues
        plot_MP_clean(eigenvalues_clean,lambdas,P0s,axs[1][1])
        return C_clipped


    


def log_return_matrix(SnP_dir, chunk_size=50, output_dir="log returns",
                      tar_filename="log_returns.tar", aggregation_window='1h'):
    """
    Calculates the log return matrix for a subset of tickers from trade data files.

    This function processes trade data for the stocks in a given directory (`SnP_dir`), calculates
    log returns based on trade prices, and creates a matrix where each column corresponds to the log return of
    a specific ticker. The resulting data is aggregated to a specified time granularity (default: hourly),
    with missing data (nulls) dropped, and then joined on a common timestamp index.

    Args:
        SnP_dir (str): The directory path containing the `trade` subfolder, which holds the trade data files 
                       for different tickers in `.tar` format. These files are expected to contain at least 
                       a `trade-price` column and a `timestamp` index.
        chunk_size (int): Number of tickers to process in each chunk (default: 50).
        output_dir (str): Directory to save intermediate Parquet files (default: "log returns").
        tar_filename (str): Name of the compressed output file (default: "log_returns.tar").
        aggregation_window (str): Time window for aggregation (default: '1h').

    Returns:
        None
    """

    # Ensure the output directory exists, creating it if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
      
    # Extract the list of tickers from the trade directory
    tickers = [tick.split('.tar')[0] for tick in os.listdir(SnP_dir + "/trade")]

    # Process tickers in chunks
    for i in range(0, len(tickers), chunk_size):
        res = None  # Initialize an empty result DataFrame for this chunk
        
        # Process each ticker in the current chunk
        for ticker in tickers[i:i + chunk_size]:
            # Derive the stock name by splitting the ticker string at '_05'
            cur_stock_name = ticker.split('_05')[0]

            # Load the trade data for the current ticker
            df_trade = load_trade_file(ticker)

            # Calculate the log returns for the ticker: log(price(t-1) / price(t))
            log_returns = df_trade.with_columns(
                (np.log(pl.col('trade-price').shift(1) / pl.col('trade-price')))
                .cast(pl.Float64)
                .alias(cur_stock_name)
            ).drop_nulls()  # Drop rows with null values in the log returns

            # Select the timestamp ('index') and the log return for the current ticker
            cur_res = log_returns.select(pl.col('index'), pl.col(cur_stock_name))

            # Truncate the timestamp to the nearest hour (or specified aggregation window)
            cur_res = cur_res.with_columns(pl.col('index').dt.truncate(aggregation_window))

            # Aggregate the log return by the specified window (e.g., hourly mean)
            cur_res = cur_res.with_columns(pl.col(cur_stock_name).mean().over('index'))

            # Remove duplicate entries to ensure one row per timestamp
            cur_res = cur_res.unique()

            # Merge the current ticker's data into the result DataFrame for the chunk
            if res is None:
                res = cur_res.sort('index')  # Initialize with the first ticker's data
            else:
                res = res.join(cur_res, how='inner', on='index').sort('index')  # Merge by 'index'

        # Save the processed chunk to a Parquet file
        chunk_file_path = os.path.join(output_dir, f"log_returns_chunk_{i // chunk_size + 1}.parquet")
        res.collect().write_parquet(chunk_file_path)
        print(f"Saved chunk {i // chunk_size + 1} to {chunk_file_path}")

    # Compress the entire output directory into a .tar file
    with tarfile.open(tar_filename, "w") as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    print(f"Compressed the output directory to {tar_filename}.")


def reconstruct_returns(output_dir, min_row_requirement=100):
    """
    Reconstructs the return matrix by joining Parquet files stored in the specified directory.

    This function iterates through Parquet files in the given output directory, loads each file into a 
    Polars DataFrame, and joins the data based on a common 'index' (timestamp). Only files with 
    a minimum number of rows (`min_row_requirement`) are included in the reconstruction. 

    Args:
        output_dir (str): Directory containing the Parquet files with return data.
        min_row_requirement (int): Minimum number of rows required in a Parquet file for it to 
                                   be included in the reconstruction (default: 100).

    Returns:
        pl.DataFrame: A Polars DataFrame containing the reconstructed return matrix, where each column 
                      represents the log returns for a specific ticker, and rows correspond to common timestamps.
    """
    res = None  # Initialize an empty result DataFrame

    # Iterate through all files in the output directory
    for pq in os.listdir(output_dir):
        # Read the current Parquet file into a Polars DataFrame
        cur_df = pl.read_parquet(output_dir + "/" + pq)

        # If 'res' is None, initialize it with the first DataFrame
        if res is None:
            res = cur_df
        else:
            # Check if the current DataFrame meets the minimum row requirement
            if len(cur_df) > min_row_requirement:
                # Join the current DataFrame with the result DataFrame on 'index'
                res = res.join(cur_df, on='index', how='inner')

    # Print the dimensions of the reconstructed return matrix
    print(f"Total return matrix has dimensions {res.shape}")

    return res
