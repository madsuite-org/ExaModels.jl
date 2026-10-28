# ExaModelsIO file format, version 1.1

Version 1.1 adds buffered subexpressions (Section 4.2) to 1.0. Every 1.0 file is a 1.1 file.

This document defines the file format written by `write_model` and read by
`read_model`. A file describes a nonlinear program

    minimize (or maximize)  f(x; θ)
    subject to              lcon ≤ g(x; θ) ≤ ucon,   lvar ≤ x ≤ uvar,

where `x ∈ R^nvar` are the variables and `θ ∈ R^npar` the parameters, in the
block form ExaModels uses: the objective is a sum of objective blocks, and the
constraint vector is filled by constraint blocks. Each block is one expression
evaluated over a collection of data elements.

The format describes the model, not the internal data structures of any
ExaModels version. A reader may rebuild the model in any way that evaluates the
same functions.

## 1. Container

A file is an HDF5 file with these members.

| Path | Kind | Contents |
|:--|:--|:--|
| `/` attribute `format` | string | `"ExaModelsIO"` |
| `/` attribute `format_version` | string | `"<major>.<minor>"`, here `"1.1"` |
| `/` attribute `examodels_version` | string | ExaModels version of the writer (informational) |
| `/structure` | string dataset | the structure document, UTF-8 JSON (Section 2) |
| `/arrays/x0`, `lvar`, `uvar` | 1-D datasets, length `nvar` | starting point and variable bounds |
| `/arrays/y0`, `lcon`, `ucon` | 1-D datasets, length `ncon` | starting multipliers and constraint bounds |
| `/arrays/theta` | 1-D dataset, length `npar` | the θ vector: parameter values, and the last computed values of buffered subexpressions (Section 4.2) |
| `/data/<block>/...` | datasets | data of each block (Section 4) |
| `/solution/...` | datasets, optional | solver output (Section 7) |

All arrays use the element type `eltype` (Section 2) except data columns, which
keep their own types. Infinite bounds are stored as IEEE infinities.

## 2. Structure document

A JSON object with these keys.

| Key | Value |
|:--|:--|
| `eltype` | number type name of the model (Section 5.1) |
| `name` | model name (string) |
| `minimize` | `true` to minimize, `false` to maximize |
| `nvar`, `ncon`, `npar` | counts of variables, constraints, and θ entries (parameters plus buffered subexpression values) |
| `extensions` | names of the user functions the expressions use (Section 6.2), sorted |
| `variables` | variable blocks (Section 3) |
| `parameters` | parameter blocks (Section 3) |
| `subexpressions` | buffered subexpression blocks (Section 4.2); added in 1.1, absent means none |
| `objectives` | objective blocks (Section 4) |
| `constraints` | constraint and augmentation blocks, in the order they are applied (Section 4) |

Readers must ignore keys they do not know (Section 8).

## 3. Variable and parameter blocks

`x` and `θ` are partitioned into consecutive blocks. Each block is

    {"name": <string or null>, "size": [<dim>, ...], "offset": <integer>}

A `dim` is an integer `n` (axis `1:n`) or a pair `[a, b]` (axis `a:b`). The block
holds `prod(lengths)` entries starting after `offset` entries of the vector, in
column-major order over its axes. Blocks are listed in increasing `offset`, they
do not overlap. Variable blocks cover `x`; parameter blocks and buffered subexpression blocks (Section 4.2) together cover `θ`. `name` is the name the model gives the
block, if any.

Expressions refer to variables and parameters by **global 1-based index**
(Section 6). A reader that rebuilds the model must therefore place every block at
the stored `offset`, and must fail if it cannot.

## 4. Objective and constraint blocks

Every block has

| Key | Value |
|:--|:--|
| `group` | HDF5 group holding the block's data, e.g. `"data/con_3"` |
| `iterator` | the collection the block iterates over (Section 4.1) |
| `expr` | expression evaluated for each element `p` of the iterator (Section 6), or null |

The objective is the sum of `expr(p)` over all elements of all objective blocks.

A constraint block additionally has `"kind": "constraint"`, `name` (string or
null), `offset` (integer) and `shape` (list of integers). It defines rows
`offset+1 … offset+prod(shape)` of `g`; row `offset + k` is `expr(p_k)` for the
k-th element in iteration order. If `expr` is null the block also has `size`
(list of `dim`, as in Section 3) and its rows start at zero; they receive values
only from augmentations.

An augmentation block has `"kind": "augmentation"`, `base_offset` (the `offset`
of an earlier constraint block, its *base*) and `index`. For every element `p`
it adds `expr(p)` to one row of the base:

* `index` null: the k-th element adds to row `base_offset + k`.
* `index` an expression evaluating to an integer `r`: row `base_offset + r`.
* `index` a `tuple` expression evaluating to `(c_1, …, c_d)`: row
  `base_offset + 1 + Σ_j (c_j − 1) Π_{l<j} n_l`, where `n_l` are the lengths of the
  base block's axes (column-major, 1-based coordinates).

### 4.1 Iterators

`iterator` is one of

* `{"collection": <data schema>}`: the elements of the stored collection
  (Section 5.2), in column-major order.
* `{"product": [<data schema>, ...]}`: the Cartesian product of the listed axes;
  the element is the tuple `(a_1, …, a_d)`, and the first axis varies fastest.
  Axis `i` is stored under `<group>/axis_<i>`. Used for constraint blocks written as
  `for a_1 in A_1, …, a_d in A_d`; the block's `shape` is the axis lengths.

### 4.2 Buffered subexpressions (1.1)

A buffered subexpression block holds values that other expressions read, so a
shared term is evaluated once instead of being repeated at every use. It has
`group`, `iterator` and `expr` as above, plus `offset` (integer) and `name`
(string or null). Element `k` of its iterator defines

    θ[offset + k] = expr(p_k),

and expressions read it with a `sub` node (Section 6). Parameter blocks and
buffered subexpression blocks share θ: they occupy consecutive, non-overlapping
ranges, and a reader recreates them in increasing `offset`. A buffered block may
read parameters and buffered values at lower offsets only. The stored values of
these θ entries are informational: a reader recomputes them from `x`.

## 5. Numbers and data

### 5.1 Numbers

Type names: `Float64`, `Float32`, `Float16`, `Int64`, `Int32`, `Int16`, `Int8`,
`UInt64`, `UInt32`, `UInt16`, `UInt8`, `Bool`.

A number in the structure document is a pair `[<type name>, <value>]`. For an
integer type the value is a JSON integer, for `Bool` a JSON boolean. For a
floating-point type it is the value converted to binary64 (exact for all listed
types) written as a JSON number that reads back to the same binary64 value, or one
of the strings `"inf"`, `"-inf"`, `"nan"`.

### 5.2 Data schemas

A data schema describes a collection stored as columns.

* `{"range": [a, b]}`: the integers `a, a+1, …, b`; nothing is stored.
* `{"array": [d_1, …], "elem": <element schema>}`: an array with dimensions
  `d_1, …`; its elements, in column-major order, are given by the element schema.

Element schemas, where `<path>` is the column's path relative to the block's
group (or axis group):

* `{"scalar": <type name>}`: one 1-D dataset at `<path>`, or at `value` when the
  path is empty.
* `{"namedtuple": [[<field name>, <element schema>], ...]}`: a record; field `f` is
  stored under `<path>/f`.
* `{"tuple": [<element schema>, ...]}`: a tuple; position `i` (1-based) is stored
  under `<path>/<i>`.

All columns of one collection have the same length. Records written from any
struct type are read back as named tuples with the same fields.

## 6. Expressions

An expression node is a JSON object with exactly one key:

| Node | Meaning |
|:--|:--|
| `{"op": [<operator>, a_1, …]}` | the operator applied to the arguments |
| `{"sum": [a_1, …]}` | `a_1 + … + a_n` |
| `{"prod": [a_1, …]}` | `a_1 * … * a_n` |
| `{"var": i}` | `x[i]`; `i` is an expression evaluating to an integer |
| `{"par": i}` | `θ[i]`; as above |
| `{"sub": i}` | the buffered subexpression value `θ[i]` (1.1); as above |
| `{"data": [s_1, …]}` | the element `p`, then field `s_j` (a string) or position `s_j` (an integer, 1-based) of the previous result, for each `s_j` in turn; `[]` is `p` itself |
| `{"const": <number>}` | a number |
| `{"null": <number or null>}` | a constant term; null means 0 |
| `{"static": <number>}` | a number |
| `{"val": <number>}` | an exponent `k`: `{"op": ["^", a, {"val": k}]}` is `a^k` |
| `{"tuple": [a_1, …]}` | a tuple; used only as an augmentation `index` |

`null`, `static` and `val` evaluate exactly like `const`; they tell an ExaModels
reader which node type to rebuild. Index arithmetic (inside `var`, `par` and
`index`) is integer arithmetic.

### 6.1 Operator vocabulary

Each operator is the Julia function of the same name, with Julia's semantics.

* From Julia `Base`, unary: `inv sqrt cbrt abs abs2 sign exp exp2 exp10 expm1 log
  log2 log1p log10 sin cos tan asin acos atan acot csc sec cot sinh cosh tanh
  asinh acosh csch sech coth sind cosd tand cscd secd cotd atand acotd sinpi cospi
  sinc deg2rad rad2deg signbit floor ceil atanh acoth`.
* From Julia `Base`, unary or binary: `+ -`.
* From Julia `Base`, binary: `* / ^ hypot max min mod`, and `atan` (two arguments).
* `bselect(s, x)`: `s * x`, except that it is 0 when `s` is 0 even if `x` is not
  finite.
* From SpecialFunctions.jl, unary: `airyai airyaiprime airybi airybiprime besselj0
  besselj1 bessely0 bessely1 dawson digamma erf erfc erfcinv erfcx erfi erfinv gamma
  invdigamma trigamma`; binary: `beta logbeta`.

Names are never removed from this list and never change meaning. New names may
be added in a minor version.

### 6.2 User functions

A function outside the vocabulary is written `"ext:<name>"`, where `<name>` is
the Julia module path and function name (e.g. `"ext:Main.myf"`), and `<name>` is
listed in `extensions`. The file does not contain its definition or derivatives.
A reader must be given the function by name and must fail before building the
model if any listed extension is missing.

## 7. Solution

`/solution` is optional. When present it may contain `solution` (length `nvar`),
`multipliers` (length `ncon`), `multipliers_L` and `multipliers_U` (length `nvar`),
`objective` (scalar) and `status` (string), with the meanings of the
JuliaSmoothOptimizers `GenericExecutionStats` fields of the same names.

## 8. Versioning

`format_version` is `<major>.<minor>`.

* A minor version only adds: new keys, new operators, new node or schema kinds.
  A file of version `M.n` is a valid file of version `M.m` for `m > n`.
* A reader for major version `M` reads every `M.x` file. It ignores keys it does
  not know, and fails on an operator, node kind or schema kind it does not know.
* Any other change increases the major version, and a reader refuses files of
  another major version.

## 9. Reader obligations

A reader must fail, rather than return a model, when

* the file has buffered subexpressions and the reader cannot build them;

* `format` is not `"ExaModelsIO"` or the major version differs;
* an extension listed in `extensions` is not supplied;
* a variable, parameter or constraint block cannot be placed at its stored
  `offset`, or the rebuilt model's `nvar`, `ncon`, `npar` differ from the stored ones;
* the columns of one collection differ in length.
