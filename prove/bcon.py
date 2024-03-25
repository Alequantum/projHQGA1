from baseconv import BaseConverter


def get_base_converter(base):
    """
    Genera un convertitore per una base numerica specificata.

    Args:
        base (int): La base numerica per cui generare il convertitore.

    Returns:
        BaseConverter: Un convertitore per la base specificata.
    """
    # Genera la stringa dei simboli per la base specificata
    # Assumendo che per base > 10 vengano usati lettere dell'alfabeto
    symbols = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    base_symbols = symbols[:base]

    # Crea e restituisce il convertitore per la base specificata
    return BaseConverter(base_symbols)


# Esempio di utilizzo:
base = 3  # Definisci la base desiderata
base_conv = get_base_converter(base)

num_in_base10 = 5
num_in_baseX = base_conv.encode(num_in_base10)
print(f"Numero in base {base}: {num_in_baseX}")

num_in_baseX = '12'
num_in_base10 = base_conv.decode(num_in_baseX)
print(f"Numero in base 10: {num_in_base10}")