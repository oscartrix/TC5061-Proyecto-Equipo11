"""Contratos de columnas de los datos de MiBici"""

# Contrato de archivos de viaje (mes)
# Nombre en el CSV de MiBici -> nombre canónico
TRIP_RENAME = {
    'Viaje_Id': 'trip_id',
    'Usuario_Id': 'user_id',
    'Genero': 'gender',
    'Año_de_nacimiento': 'birth_year',
    'Inicio_del_viaje': 'start_time',
    'Fin_del_viaje': 'end_time',
    'Origen_Id': 'origin_id',
    'Destino_Id': 'destination_id',
}
TRIP_SOURCE_COLUMNS = tuple(TRIP_RENAME)
TRIP_COLUMNS = tuple(TRIP_RENAME.values())

# Variantes detectadas
TRIP_HEADER_VARIANTS = {
    'Año_de_nacimiento': {'A}äe_nacimiento'},
}

# Columnas con datos personales
TRIP_PERSONAL_COLUMNS = ('user_id', 'gender', 'birth_year')

TRIP_DTYPES = {'trip_id': 'int64', 'origin_id': 'Int32', 'destination_id': 'Int32'}
TRIP_DATETIME_COLUMNS = ('start_time', 'end_time')

# Formatos de fecha detectados
TRIP_DATETIME_FORMATS = (
    '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M',
)

# Contrato del catálogo de estaciones
# Nombre en el CSV de MiBici -> nombre canónico
STATION_RENAME = {
    'id': 'station_id',
    'name': 'name',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'dpcapacity': 'capacity',
}
STATION_SOURCE_COLUMNS = tuple(STATION_RENAME)
STATION_COLUMNS = tuple(STATION_RENAME.values())

# No hay variantes del encabezado de estaciones
STATION_HEADER_VARIANTS: dict[str, set[str]] = {}

STATION_DTYPES = {
    'station_id': 'Int32',
    'name': 'string',
    'latitude': 'float64',
    'longitude': 'float64',
    'capacity': 'Int32',
}
