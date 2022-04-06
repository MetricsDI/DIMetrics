"""
Simple JSON Parser
==================

The code is short and clear, and outperforms every other parser (that's written in Python).
For an explanation, check out the JSON parser tutorial at /docs/json_tutorial.md
"""
import sys

from lark import Lark, Transformer, v_args

json_grammar = r"""
    ?start: value

    ?value: object
          | array
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : string ":" value

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS
"""


class TreeToJson(Transformer):
    @v_args(inline=True)
    def string(self, s):
        return s[1:-1].replace('\\"', '"')

    array = list
    pair = tuple
    object = dict
    number = v_args(inline=True)(float)

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False


### Create the JSON parser with Lark, using the Earley algorithm
# json_parser = Lark(json_grammar, parser='earley', lexer='standard')
# def parse(x):
#     return TreeToJson().transform(json_parser.parse(x))

### Create the JSON parser with Lark, using the LALR algorithm
json_parser = Lark(json_grammar, parser='lalr',
                   # Using the standard lexer isn't required, and isn't usually recommended.
                   # But, it's good enough for JSON, and it's slightly faster.
                   lexer='standard',
                   # Disabling propagate_positions and placeholders slightly improves speed
                   propagate_positions=False,
                   maybe_placeholders=False,
                   # Using an internal transformer is faster and more memory efficient
                   transformer=TreeToJson())
parse = json_parser.parse


def test():
    #test tax json
    test_json = '''
        {
    "NjTyEndDate": ",",
    "NjPartnerEin": "13-3806691",
    "NjPartnerName": "BLACKROCK FINANCIAL MANAGEMENT, INC.",
    "NjPartnerAddress": "55 EAST 52ND STREET",
    "NjPartnerCity": "NEW YORK,,",
    "NjPartnerState": "NY",
    "NjPartnerZip": "10055",
    "NjPartnerEntityType": "FC",
    "NjPartnerInterestPship": "01/20/2015",
    "NjPartnershipEin": "46-5515706",
    "NjPartnershipName": "LUMINEX TRADING & ANALYTICS LLC",
    "NjPartnershipAddress": "157 SEAPORT BLVD., SUITE P3",
    "NjPartnershipCity": "BOSTON,",
    "NjPartnershipState": "MA",
    "NjPartnershipZip": "02110",
    "NjProfitSharingI": "4.876069 % 4.8689968",
    "NjProfitSharingIi": "%",
    "NjLossSharingI": "4.876069 % 4.8689968",
    "NjLossSharingIi": "%",
    "NjCapitalOwnershipI": "4.900000 % 4.8999946",
    "NjCapitalOwnershipIi": "%",
    "NjPartIiPshipIncomeA": "-252,867.",
    "NjPartIiPshipIncomeB": "-26,969.",
    "NjPartIiDistributiveShareA": "-252,867.",
    "NjPartIiDistributiveShareB": "-26,969."
}

    '''

    j = parse(test_json)
    for key in j:
        print(f'{key} --> {j[key]}')
    import json
    assert j == json.loads(test_json)

def parse_json(json_path):
    with open(json_path) as f:
        json_dict = parse(f.read())
    return json_dict


if __name__ == '__main__':
    test()
