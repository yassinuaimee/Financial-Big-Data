import polars as pl
import numpy as np
import pandas as pd
import os
import tarfile
import io

def cast_to_common_dtypes(lazy_frame, most_common_data_types):
    cast_dict = {
                'Int8': pl.Int8,
                'Int16': pl.Int16,
                'Int32': pl.Int32,
                'Int64': pl.Int64,
                'UInt8': pl.UInt8,
                'UInt16': pl.UInt16,
                'UInt32': pl.UInt32,
                'UInt64': pl.UInt64,
                'Float32': pl.Float32,
                'Float64': pl.Float64,
                'Utf8': pl.Utf8,
                'Date': pl.Date,
                'Time': pl.Time,
                'Duration': pl.Duration,
                'Categorical': pl.Categorical,
                'Boolean': pl.Boolean,
                'Object': pl.Object,
                'Struct': pl.Struct,
                'List': pl.List,
                'Binary': pl.Binary,
            }
    schema = lazy_frame.schema
    for item, ref_type in zip(schema.items(),most_common_data_types):
        if str(item[1]) != str(ref_type):
            lazy_frame = lazy_frame.with_columns(
                    pl.col(item[0]).cast(cast_dict[str(ref_type)]).alias(item[0])
                )
    return lazy_frame


def wrangle_trade(DF,
            tz_exchange="America/New_York",
            only_non_special_trades=True,
            only_regular_trading_hours=True,
            hhmmss_open="09:30:00",
            hhmmss_close="16:00:00",
            merge_sub_trades=True):
    excel_base_date = pl.datetime(1899, 12, 30)  # Excel starts counting from 1900-01-01, but Polars needs 1899-12-30
    DF = DF.with_columns(
        (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("index")
    )
    DF = DF.with_columns(pl.col("index").dt.convert_time_zone(tz_exchange))

    if only_non_special_trades:
        DF=DF.filter(pl.col("trade-stringflag")=="uncategorized")

    DF = DF.drop(["xltime","trade-rawflag","trade-stringflag"])

    if only_regular_trading_hours:
        hh_open,mm_open,ss_open = [int(x) for x in hhmmss_open.split(":")]
        hh_close,mm_close,ss_close = [int(x) for x in hhmmss_close.split(":")]

        seconds_open=hh_open*3600+mm_open*60+ss_open
        seconds_close=hh_close*3600+mm_close*60+ss_close

        DF = DF.filter(pl.col('index').dt.hour().cast(pl.Int32)*3600+pl.col('index').dt.minute().cast(pl.Int32)*60+pl.col('index').dt.second().cast(pl.Int32)>=seconds_open,
                       pl.col('index').dt.hour().cast(pl.Int32)*3600+pl.col('index').dt.minute().cast(pl.Int32)*60+pl.col('index').dt.second().cast(pl.Int32)<=seconds_close)


    if merge_sub_trades:   # average volume-weighted trade price here
        DF=DF.group_by('index',maintain_order=True).agg([(pl.col('trade-price')*pl.col('trade-volume')).sum()/(pl.col('trade-volume').sum()).alias('trade-price'),pl.sum('trade-volume')])        
    return DF


def load_trade_file(ticker,
            tz_exchange="America/New_York",
            only_non_special_trades=True,
            only_regular_trading_hours=True,
            merge_sub_trades=True):
    
    raw_dir="SP500_2010/SP500_2010/trade/"+ticker+".tar"

    lazy_frames = []
    all_dtypes=[]
    with tarfile.open(raw_dir, 'r') as tar:
        for member in tar.getmembers():
            # Check if the file is a Parquet file
            if member.name.endswith(".parquet"):
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    buffer = io.BytesIO(file_obj.read())
                    lazy_frame = pl.scan_parquet(buffer)
                    lazy_frames.append(lazy_frame)
                    dtypes = lazy_frame.dtypes
                    all_dtypes.append(dtypes)

    dtypes_DF = pd.DataFrame(all_dtypes)
    dtypes_DF = dtypes_DF.astype(str)
    most_common_data_types = pd.DataFrame(dtypes_DF.value_counts()).iloc[0].name

    if lazy_frames:
        DF=pl.concat([cast_to_common_dtypes(l,most_common_data_types) for l in lazy_frames])
    
    DF = wrangle_trade(DF,
            tz_exchange=tz_exchange,
            only_non_special_trades=only_non_special_trades,
            only_regular_trading_hours=only_regular_trading_hours,
            merge_sub_trades=merge_sub_trades)

    return DF

def wrangle_bbo(DF,
            tz_exchange="America/New_York",
            only_regular_trading_hours=True,
            hhmmss_open="09:30:00",
            hhmmss_close="16:00:00",
            merge_same_index=True):

    excel_base_date = pl.datetime(1899, 12, 30)  # Excel starts counting from 1900-01-01, but Polars needs 1899-12-30
    DF = DF.with_columns(
        (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("index")
    )
    DF = DF.with_columns(pl.col("index").dt.convert_time_zone(tz_exchange))
    DF = DF.drop("xltime")

    # apply common sense filter
    DF = DF.filter(pl.col("ask-price")>0).filter(pl.col("bid-price")>0).filter(pl.col("ask-price")>pl.col("bid-price"))

    if merge_same_index:
        DF = DF.group_by('index',maintain_order=True).last()   # last quote of the same timestamp
    
    if only_regular_trading_hours:
        hh_open,mm_open,ss_open = [int(x) for x in hhmmss_open.split(":")]
        hh_close,mm_close,ss_close = [int(x) for x in hhmmss_close.split(":")]

        seconds_open=hh_open*3600+mm_open*60+ss_open
        seconds_close=hh_close*3600+mm_close*60+ss_close

        DF = DF.filter(pl.col('index').dt.hour().cast(pl.Int32)*3600+pl.col('index').dt.minute().cast(pl.Int32)*60+pl.col('index').dt.second().cast(pl.Int32)>=seconds_open,
                       pl.col('index').dt.hour().cast(pl.Int32)*3600+pl.col('index').dt.minute().cast(pl.Int32)*60+pl.col('index').dt.second().cast(pl.Int32)<=seconds_close)
    return DF

def load_bbo_file(ticker,
            tz_exchange="America/New_York",
            only_regular_trading_hours=True,
            hhmmss_open="09:30:00",
            hhmmss_close="16:00:00",
            merge_same_index=True):
    
    raw_dir="SP500_2010/SP500_2010/bbo/"+ticker+".tar"

    lazy_frames = []
    all_dtypes=[]

    with tarfile.open(raw_dir, 'r') as tar:
        for member in tar.getmembers():
            # Check if the file is a Parquet file
            if member.name.endswith(".parquet"):
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    buffer = io.BytesIO(file_obj.read())
                    lazy_frame = pl.scan_parquet(buffer)
                    lazy_frames.append(lazy_frame)
                    dtypes = lazy_frame.dtypes
                    all_dtypes.append(dtypes)

    dtypes_DF = pd.DataFrame(all_dtypes)
    dtypes_DF = dtypes_DF.astype(str)
    most_common_data_types = pd.DataFrame(dtypes_DF.value_counts()).iloc[0].name

    if lazy_frames:
        DF=pl.concat([cast_to_common_dtypes(l,most_common_data_types) for l in lazy_frames])

    DF = wrangle_bbo(DF,
            tz_exchange=tz_exchange,
            only_regular_trading_hours=only_regular_trading_hours,
            hhmmss_open=hhmmss_open,
            hhmmss_close=hhmmss_close,
            merge_same_index=merge_same_index)

    
    return DF





