import math
import operator
from collections.abc import Iterable
from functools import cache, partial
from itertools import accumulate, combinations, product, starmap
from typing import Callable, Generator, Generic, Self, TypeVar

import pandas as pd
from pydantic import BaseModel, Field, computed_field

T = TypeVar("T", bound=int)

class Dimension(BaseModel, Generic[T]):
    name: str
    null: T
    grid: tuple[T, ...] = Field(repr=False)

    @computed_field
    @property
    def size(self) -> int:
        return len(self.grid)

    @computed_field(repr=False)
    @property
    def _index_map(self) -> dict[T, int]:
        return dict(map(reversed, enumerate(self.grid)))

    @computed_field(repr=False)
    @property
    def isnull(self) -> str:
        return f"({self.name} == {repr(self.null)})"

    @computed_field(repr=False)
    @property
    def notnull(self) -> str:
        return f"({self.name} != {repr(self.null)})"

    def index(self, value: T) -> tuple[int, ...]:
        if value == self.null:
            return tuple(range(self.size))
        return (self._index_map[value],)

    def order(self, value: T) -> int:
        if value == self.null:
            return self.size
        return self._index_map[value]

    def __hash__(self) -> int:
        return hash((self.name, self.grid, self.null))

    @classmethod
    def from_pandas_series(cls, series: pd.Series, null: T) -> Self:
        dtype = _infer_dtype(series)
        name = series.name
        null = dtype(null)
        grid = series[series != null].unique().astype(dtype).tolist()
        return cls[dtype](name=name, null=null, grid=grid)


def _infer_dtype(series: pd.Series) -> type:
    if pd.api.types.is_integer_dtype(series):
        return int
    if pd.api.types.is_string_dtype(series):
        return str
    raise TypeError("Dimension only supports 'int' or 'str' dtypes")


class Space(BaseModel):
    dimensions: tuple[Dimension, ...] = Field(repr=False)

    @computed_field
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dim.name for dim in self.dimensions)

    @computed_field
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(dim.size for dim in self.dimensions)

    @computed_field(repr=False)
    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    @computed_field(repr=False)
    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @computed_field(repr=False)
    @property
    def isnull(self) -> tuple[str, ...]:
        return tuple(dim.isnull for dim in self.dimensions)

    @computed_field(repr=False)
    @property
    def notnull(self) -> tuple[str, ...]:
        return tuple(dim.notnull for dim in self.dimensions)

    @cache
    def span(self) -> pd.DataFrame:
        data = product(*[dim.grid for dim in self.dimensions])
        return pd.DataFrame(data=data, columns=self.names)

    def index(
        self, values: tuple[int | str, ...], ravel: bool = True
    ) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        multi_index = product(
            *[dim.index(value) for dim, value in zip(self.dimensions, values)]
        )
        if not ravel:
            return tuple(multi_index)

        vec = tuple(accumulate((1, *reversed(self.shape[1:])), operator.mul))
        vec = tuple(reversed(vec))
        res = map(partial(zip, vec), multi_index)
        res = map(partial(starmap, operator.mul), res)
        return tuple(map(sum, res))

    def split(self, sizes: tuple[int, ...]) -> Iterable:
        if sum(sizes) > self.ndim:
            raise ValueError("Sum of sizes exceeds space size")

        comb = split_combinations(self.dimensions, sizes)
        return map(mapper(Space.from_dimensions), comb)

    @classmethod
    def from_dimensions(cls, dimensions: Iterable[Dimension]) -> Self:
        return cls(dimensions=dimensions)

    def __hash__(self) -> int:
        return hash(self.dimensions)


def split_combinations(
    curr_iter: Iterable, sizes: tuple[int, ...]
) -> Generator:
    if len(sizes) == 0:
        yield (tuple(curr_iter),)
    else:
        for curr_comb in combinations(curr_iter, sizes[0]):
            next_iter = list(x for x in curr_iter if x not in curr_comb)
            for next_comb in split_combinations(next_iter, sizes[1:]):
                yield curr_comb, *next_comb


def mapper(fun: Callable) -> Callable:
    def wrapper(x: Iterable) -> Iterable:
        return map(fun, x)

    return wrapper
