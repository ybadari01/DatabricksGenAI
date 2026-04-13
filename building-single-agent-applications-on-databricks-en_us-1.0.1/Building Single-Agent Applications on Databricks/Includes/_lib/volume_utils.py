from pathlib import Path
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def create_volume(
    catalog_name: str,
    schema_name: str,
    volume_name: str = "demo_vol",
) -> Path:
    """
    Create a Unity Catalog volume if it doesn't exist.

    Parameters
    ----------
    catalog_name : str
        The UC catalog name.
    schema_name : str
        The UC schema name.
    volume_name : str
        The volume name. Default: 'demo_vol'.

    Returns
    -------
    Path
        Path to the volume: /Volumes/{catalog}/{schema}/{volume}
    """
    spark.sql(f"USE CATALOG {catalog_name}")
    spark.sql(f"USE SCHEMA {schema_name}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}")

    volume_path = Path(f"/Volumes/{catalog_name}/{schema_name}/{volume_name}")
    print(f"  Volume ready: {volume_path}")
    return volume_path
