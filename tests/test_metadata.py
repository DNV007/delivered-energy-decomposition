from src.metadata import extract_cell_id, extract_temperature_deg_c, infer_cell_family, infer_chemistry


def test_cell_metadata_from_temperature_filename():
    path = "raw_CU_diffT/TC23SIB09/TC23SIB09_05deg.parquet"
    assert extract_cell_id(path) == "TC23SIB09"
    assert infer_cell_family(path) == "SIB"
    assert infer_chemistry(path) == "sodium-ion"
    assert extract_temperature_deg_c(path) == 5


def test_lithium_reference_family_metadata():
    path = "raw_EIS_plotting/TC23NMC04/auxiliary/20231113_TC23NMC04_3900mV_25gradC_EIS.csv"
    assert extract_cell_id(path) == "TC23NMC04"
    assert infer_cell_family(path) == "NMC"
    assert infer_chemistry(path) == "lithium-ion"
    assert extract_temperature_deg_c(path) == 25
