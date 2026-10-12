---
name: grilling
description: Asking about an underspecified request so the answers are worth having. Adapted from Matt Pocock's grilling skill.
# No triggers: this is not situational reference, it is how the one phase whose
# entire job is asking should ask. An empty trigger list means always.
triggers: []
phases: [clarify]
---

A question costs the person's attention and the run's budget, and this round may
ask five at most, all shown together. What makes them worth having is not how
many unknowns they cover but whether each one can actually be answered, and
whether the answers still make sense together.

## Facts are your job; decisions are theirs

Never ask where something lives in the code, what a function currently does,
whether a file exists, or which version is installed. You have the workspace
as this message shows it; the person does not have the code in their head. A traced run asked "is the colour change happening in the Bird
class `update()` method or elsewhere?" and was told, correctly, "I don't know
where it is occurring in the code, I am observing the behaviour." That question
spent one of the round's questions and returned nothing.

Ask only what is genuinely theirs: what they want, which of two designs, what
the goal did not settle, what should happen in a case nobody specified.

## Ask the frontier, not the tree

Think of the open decisions as a tree: some hang off others. The **frontier**
is the decisions whose prerequisites are already settled — the ones you can ask
about *now* without guessing at an answer you have not heard yet.

Ask the frontier. Leave anything downstream of an open question for later,
because a question that only makes sense once another is answered will be
answered against an assumption the person never made. One run asked what
behaviour clouds should have — "static background" — and separately whether
clouds should collide with the bird — "collision". Both answered honestly, and
they contradict, because the second question belonged to a round that had not
happened yet.

If the frontier holds only one real question, ask one. Three shallow questions
are worse than one that matters.

## Make each question answerable in a word

Offer the concrete options inside the question, and say which you would pick.
"Should the cache be per-process, or shared via Redis?" can be answered with a
click; "What are your caching requirements?" starts an essay. A recommendation
is not presumption — it gives them something to disagree with, which is far
easier than composing an answer from nothing.

State what each answer would cause you to do, when that is not obvious. The
person is deciding between outcomes, not between words.

## Read the answer against the question you asked

A short answer is only meaningful next to what was asked. "Neither", "the first
one", "no" — none of these carry their meaning alone, and an answer that
rejects both offered options rejects them *both*; it does not pick the nearer
one. Where an answer contradicts the goal's wording, the answer wins: it was
written second, in response to being asked precisely this.

If an answer leaves you no wiser, say so in the plan rather than quietly
picking. An assumption you name can be corrected; one you bury cannot.
