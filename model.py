class Field:
    def __init__(self, name, ftype):
        self.name = name
        self.ftype = ftype  

class CollectionSchema:
    def __init__(self, name, json_schema):
        self.name = name
        self.fields = self._parse(json_schema["properties"])

    def _parse(self, properties):
        fields = []
        for name, prop in properties.items():
            fields.append(Field(name, prop["type"]))
        return fields
