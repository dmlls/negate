<p align="center"><img width="666" src="https://user-images.githubusercontent.com/22967053/208313993-b873be19-8648-4edd-b4ee-e5283c19ad8f.png" alt="Negate: A Python module to negate sentences."></p>
<p align="center" display="inline-block">
  <a href="https://pypi.org/project/negate/">
    <img src="https://img.shields.io/pypi/v/negate">
  </a>
  <a href="https://pypi.org/project/negate/">
    <img src="https://img.shields.io/badge/release-beta-green">
  </a>
  <a href="https://deepsource.io/gh/dmlls/negate/?ref=repository-badge" target="_blank">
      <img alt="Active Issues" title="DeepSource" src="https://deepsource.io/gh/dmlls/negate.svg/?label=active+issues&token=D9QCfE028iloctbSdOhywtZy"/>
  </a>
</p>

<br>

## Introduction

Negate is a Python module that implements rule-based, syntactic sentence
negation in English<sup>1</sup>.


<sub><sup>1</sup> More languages might be supported in the future.</small></sub>

<br>

## Installation

Negate is available on [PyPI](https://pypi.org/project/negate/) and can be
installed using pip:
```shell
pip install -U negate
```

<br>

## Usage

### 1. Initializing the Negator

First the Negator must be initialized:

```Python
from negate import Negator

# Use default model (en_core_web_md):
negator = Negator()

```

<br>

By default, negate uses the spaCy model
[`en_core_web_md`](https://spacy.io/models/en#en_core_web_md) for POS tagging
and dependency parsing. However, in cases where accuracy is preferred over
efficiency, negate also allows to use a Transformer model, namely the spaCy
model [`en_core_web_trf`](https://spacy.io/models/en#en_core_web_trf). To use
this model, simply initialize the Negator passing `use_transformers=True`:

```Python
# Or use a Transformer model (en_core_web_trf):
negator = Negator(use_transformers=True)

# Use a Transformer model with GPU (if available):
negator = Negator(use_transformers=True, use_gpu=True)
```

If the models are not locally installed, negate will download them and install
them first. This only needs to occur once.

<br>

### 2. Negating sentences

Then, to negate a sentence:

```Python
sentence = "An apple a day, keeps the doctor away."

negated_sentence = negator.negate_sentence(sentence)

print(negated_sentence)  # "An apple a day, doesn't keep the doctor away."
```

<br>

When the parameter `prefer_contractions` is set to `True` (default),
modifications to auxiliary verbs will use their contracted form<sup>2</sup>. For
example:

```Python
sentence = "Speaking of doctors, I went to the doctor the other day."

negated_sentence = negator.negate_sentence(sentence, prefer_contractions=True)
print(negated_sentence)  # "Speaking of doctors, I didn't go to the doctor the other day."

negated_sentence = negator.negate_sentence(sentence, prefer_contractions=False)
print(negated_sentence)  # "Speaking of doctors, I did not go to the doctor the other day."
```

<sub><sup>2</sup> Note that this does not affect other existent verbs in the
sentence that haven't been modified.</small></sub>

<br>

### Behavior upon unsupported sentences

Currently, negate will not be able to negate certain types of sentences (see
[Current Limitations](#current-limitations) and [Irremediable
Caveats](#irremediable-caveats)).

In some cases, negate will detect that a sentence is not supported. By default,
a warning will be issued:

```console
Negator - WARNING: Sentence not supported. Output might be arbitrary.
```

<br>

If you want the negator to fail instead of printing a warning, simply initialize
it with `fail_on_unsupported` set to `True`, i.e.:

```Python
negator = Negator(fail_on_unsupported=True)
```

<br>

This can be useful to skip unsupported sentences when running negate on a batch
of sentences, e.g.:

```Python
negator = Negator(fail_on_unsupported=True)
sentences = [...]
negated_sentences = []

for sent in sentences:
    try:
        negated_sentences.append(negator.negate_sentence(sent))
    except RuntimeError:
        pass  # skip unsupported sentence
```

<br>

## Current Limitations

**Negate should work fine for most cases.** However, it is currently in beta
phase. Some features have not yet been implemented. Pull Requests are always
welcome!


- [ ] **"Some", "any", "yet" are not properly supported.** E.g.: Negating the
  sentence "There are some features to be implemented." will currently output
  "There aren't some features to be implemented." Although this could still make
  sense depending on the context (e.g., "There aren't some features to be
  implemented. No, not just *some*, there are a lot!"), I assume most users
  would expect "some" being replaced with "any" and vice versa. When it comes to
  "yet", when negating a negation, it makes sense to remove it or replace it
  with "already", e.g., "I haven't been to Paris yet." → "I have been to Paris."

- [ ] **[Inversions](https://dictionary.cambridge.org/es-LA/grammar/british-grammar/inversion)
  are not supported.** This mainly affects to questions, e.g., "*Did* you go to
  the concert?" vs. "You *did* go to the concert." Notice how in the first
  example (interrogative) we have AUX + PRON + VERB and in the second
  (affirmative) PRON + AUX + VERB.

- [ ] **Non-verbal negations are not supported.** This type of negations, such
  as "A bottle with no cap." will produce the warning: `Negator - WARNING:
  Sentence not supported. Output might be arbitrary`.

- [ ] **The auxiliary "ought" is not supported.** "Ought" is the only auxiliary
  followed by a "to." This complicates things slightly. But yeah, it ought to be
  implemented at some point.

- [ ] **Certain verb conjunctions are not supported.** E.g.: "She hates and
  loves winter." → "She doesn't *hate* and *love* winter." Currently, only the
  first verb will be correctly conjugated. In this cases, it would also make
  sense to attend to boolean algebra (De Morgan's law) and replace the "and"
  with "or"/ "neither"/"nor", i.e., "She doesn't *hate* nor *love* winter."

- [ ] **Multiple verb negation is not supported.** In many sentences with
  subordinate clauses, it would make sense to negate several verbs. E.g.: "I am
  hungry because I didn't eat." → "I am *not* hungry because I *ate*."

<br>

## Irremediable Caveats

Language took very seriously Bruce Lee's famous words "be water, my friend." It
is extremely flexible, and therefore, no number of rules, however large this
number may be, will cover the whole realm of possibilities. Just when you think
your rules cover most of the cases, a new one comes in that breaks things. Early
NLP researchers and developers know very well about this.

Negate has no notion of meaning neither will ever do. Its scope its limited to
syntax. Because of this, in some cases, the produced negated sentences might
sound rather off.

Negate depends 100 % on POS tagging and dependency parsing. If any of them
fails, negate will also fail. The spaCy models we use are not infallible, which
adds another layer of "things that could go wrong" to negate.

This module was in fact developed to generate negation data in order to
fine-tune NLP Deep Learning models (yes, Transformers – can't believe I made it
this long without mentioning the word). They are, of course, the way to go for a
fully-fledged negation that also attends to semantics.

<br>

## Acknowledgements

Negate has two core direct dependencies. Without them, negate wouldn't be able
to exist:

- [spaCy 💫](https://github.com/explosion/spaCy): As already mentioned, we rely
  on POS tagging and dependency parsing to negate sentences. spaCy makes this
  process very easy for us.

- [LemmInflect 🍋](https://github.com/bjascob/LemmInflect): Negations go far
  beyond adding or removing a negation particle. In some cases, verbs have to be
  properly conjugated (e.g., when negating verbs in third-person or in past
  simple). LemmInflect provides us with this functionality.
