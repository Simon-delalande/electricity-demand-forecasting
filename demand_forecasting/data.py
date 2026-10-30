"""English display names for the original provider's CSV schema."""

SOURCE_COLUMNS = {
    "Périmètre": "Scope", "Nature": "Data type", "Date": "Date", "Heure": "Time",
    "Date et Heure": "Timestamp", "Consommation (MW)": "Consumption (MW)",
    "Prévision J-1 (MW)": "D-1 forecast (MW)", "Prévision J (MW)": "D forecast (MW)",
    "Fioul (MW)": "Oil (MW)", "Charbon (MW)": "Coal (MW)", "Gaz (MW)": "Gas (MW)",
    "Nucléaire (MW)": "Nuclear (MW)", "Eolien (MW)": "Wind (MW)",
    "Solaire (MW)": "Solar (MW)", "Hydraulique (MW)": "Hydro (MW)",
    "Pompage (MW)": "Pumping (MW)", "Bioénergies (MW)": "Bioenergy (MW)",
    "Ech. physiques (MW)": "Physical exchanges (MW)",
    "Taux de CO2 (g/kWh)": "CO2 intensity (g/kWh)",
    "Ech. comm. Angleterre (MW)": "Commercial exchanges - England (MW)",
    "Ech. comm. Espagne (MW)": "Commercial exchanges - Spain (MW)",
    "Ech. comm. Italie (MW)": "Commercial exchanges - Italy (MW)",
    "Ech. comm. Suisse (MW)": "Commercial exchanges - Switzerland (MW)",
    "Ech. comm. Allemagne-Belgique (MW)": "Commercial exchanges - Germany/Belgium (MW)",
    "Fioul - TAC (MW)": "Oil - combustion turbines (MW)",
    "Fioul - Cogénération (MW)": "Oil - cogeneration (MW)",
    "Fioul - Autres (MW)": "Oil - other (MW)",
    "Gaz - TAC (MW)": "Gas - combustion turbines (MW)",
    "Gaz - Cogénération (MW)": "Gas - cogeneration (MW)",
    "Gaz - CCG (MW)": "Gas - combined cycle (MW)",
    "Gaz - Autres (MW)": "Gas - other (MW)",
    "Hydraulique - Fil de l'eau + éclusée (MW)": "Hydro - run-of-river and pondage (MW)",
    "Hydraulique - Lacs (MW)": "Hydro - reservoirs (MW)",
    "Hydraulique - STEP turbinage (MW)": "Hydro - pumped-storage generation (MW)",
    "Bioénergies - Déchets (MW)": "Bioenergy - waste (MW)",
    "Bioénergies - Biomasse (MW)": "Bioenergy - biomass (MW)",
    "Bioénergies - Biogaz (MW)": "Bioenergy - biogas (MW)",
}


def read_source_data(path, columns=None):
    """Keep the source file intact and expose English columns at ingestion."""
    import pandas as pd

    reverse = {english: source for source, english in SOURCE_COLUMNS.items()}
    usecols = None if columns is None else [reverse.get(column, column) for column in columns]
    data = pd.read_csv(path, sep=";", usecols=usecols, low_memory=False).rename(columns=SOURCE_COLUMNS)
    if "Data type" in data:
        data["Data type"] = data["Data type"].replace({
            "Données consolidées": "Consolidated data",
            "Données définitives": "Final data",
        })
    return data
