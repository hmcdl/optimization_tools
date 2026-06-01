import json
import typing


    
class AbstractObject():
    """
    Абстрактный объект для расчетов или оптимизации
    Бесполезная сущность, служит для обозначения модели в сигнатуре функции,
    вызывающей расчет или оптимизацию.
    """

class CachableObject(AbstractObject):
    """
    Кэшируемый объект
    !!!При создании классов-наследников избегать параметров модели, хранящихся в
    mutable объектах!!!
    Нужно быть осторожным, так как оптимизация ничего не знает о том,
    является ли переменная в модели хэшируемой или нет. 
    
    signature(self) - создание сигнатуры для хэш-таблицы
    """
    @staticmethod
    def _normalize_signature_value(value):
        if isinstance(value, float):
            return round(value, 12)
        return value

    def signature(self):
        cache_signature_fields = getattr(self, "cache_signature_fields", None)
        if cache_signature_fields is not None:
            return tuple(
                (
                    field_name,
                    self._normalize_signature_value(getattr(self, field_name)),
                )
                for field_name in cache_signature_fields
            )

        signature_list = []
        for item in self.__dict__.values():
            if isinstance(item, typing.Hashable):
                signature_list.append(self._normalize_signature_value(item))
        return tuple(signature_list)

class AbstractJsonableObject(AbstractObject):
    def dump_to_json_string(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=True,
            indent=4)
    def dump_to_json_file(self, f):
        return json.dump(
            self,
            f,
            default=lambda o: o.__dict__, 
            sort_keys=True,
            indent=4)

