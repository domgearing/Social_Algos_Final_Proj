# domain_relevance_map.py
#
# Theoretically derived prior for the Hierarchical Stochastic Persona Simulation.
# Each entry: (life_event_type, latent_variable, direction, magnitude_range, theoretical_basis)
#
# Latent variables:
#   moral_traditionalism    — abortion, gender roles, sexuality norms
#   economic_redistribution — support for government spending, unions, redistribution
#   racial_attitudes        — structural vs. individual explanations for inequality
#   institutional_trust     — confidence in government, press, science, religion
#   outgroup_affect         — generalized social trust, immigration attitudes
#
# Directions: "up" | "down" | "mixed"
# Magnitude ranges: (min_delta, max_delta) as floats on a 0–1 Beta scale
#   small:   (0.02, 0.06)
#   moderate:(0.06, 0.12)
#   large:   (0.12, 0.22)
#
# Negativity bias multiplier: negative-valence events use the upper end of their
# magnitude range; positive-valence events use the lower end. Applied at runtime.
#
# Unmodeled events produce NO latent update — this is intentional. Sparsity
# prevents spurious correlations from an overcrowded update map.

SMALL    = (0.02, 0.06)
MODERATE = (0.06, 0.12)
LARGE    = (0.12, 0.22)

# Each tuple:
# (life_event_type, latent_variable, direction, magnitude_range, theoretical_basis)

DOMAIN_RELEVANCE_MAP = [

    # -------------------------------------------------------------------------
    # RELIGION & FAITH
    # -------------------------------------------------------------------------

    (
        "religious_trauma_or_institutional_scandal",
        "moral_traditionalism",
        "down",
        LARGE,
        # Putnam & Campbell (2010): institutional betrayal is the single strongest
        # predictor of religious disaffiliation. Pew (2019) finds abuse scandals
        # accelerate exit from organized religion, especially among young adults.
    ),
    (
        "religious_trauma_or_institutional_scandal",
        "institutional_trust",
        "down",
        LARGE,
        # Tyler (2006): procedural injustice by trusted institutions produces
        # generalized trust decline beyond the specific institution.
    ),
    (
        "religious_community_provides_support_in_crisis",
        "moral_traditionalism",
        "up",
        MODERATE,
        # Putnam & Campbell (2010): positive community experience is the strongest
        # retention mechanism. Social embeddedness in congregation predicts
        # sustained orthodox belief.
    ),
    (
        "religious_conversion_or_spiritual_awakening",
        "moral_traditionalism",
        "up",
        LARGE,
        # Pew (2014) Religious Landscape Study: converts to evangelical or
        # traditionalist denominations shift more sharply on moral items than
        # lifelong members, suggesting conversion is causally potent.
    ),
    (
        "loss_of_faith_after_personal_tragedy",
        "moral_traditionalism",
        "down",
        LARGE,
        # Pew (2019): bereavement and unanswered theodicy are top self-reported
        # reasons for religious disaffiliation. Effect is larger and more durable
        # than passive drift away from religion.
    ),
    (
        "loss_of_faith_after_personal_tragedy",
        "institutional_trust",
        "down",
        SMALL,
        # Secondary effect: disillusionment with religious institution generalizes
        # weakly to other authority structures (Putnam & Campbell 2010, Ch. 4).
    ),

    # -------------------------------------------------------------------------
    # ECONOMIC EXPERIENCE
    # -------------------------------------------------------------------------

    (
        "job_loss_or_prolonged_unemployment",
        "economic_redistribution",
        "up",
        MODERATE,
        # Schlozman, Verba & Brady (2012): direct economic hardship is the most
        # consistent predictor of redistribution support across survey waves.
        # Effect is larger for unexpected job loss than anticipated transitions.
    ),
    (
        "job_loss_or_prolonged_unemployment",
        "institutional_trust",
        "down",
        MODERATE,
        # Hochschild (2016): job loss attributed to systemic failure (trade,
        # automation, corporate decision) produces institutional distrust.
        # Effect is moderated by causal attribution — personal failure framing
        # does not produce the same trust decline.
    ),
    (
        "significant_upward_mobility_or_wealth_accumulation",
        "economic_redistribution",
        "down",
        SMALL,
        # Bartels (2008): income gains shift redistribution preferences downward,
        # but effect is modest and asymmetric — losses shift preferences up more
        # than equivalent gains shift them down (loss aversion in economic attitudes).
    ),
    (
        "workplace_exploitation_or_wage_theft",
        "economic_redistribution",
        "up",
        MODERATE,
        # Schlozman et al. (2012): direct experience of labor exploitation
        # increases union support and redistribution preferences more than
        # general economic hardship, because the causal attribution is clear.
    ),
    (
        "workplace_exploitation_or_wage_theft",
        "institutional_trust",
        "down",
        MODERATE,
        # Tyler (2006): perceived procedural injustice in the workplace generalizes
        # to distrust of legal and regulatory institutions perceived as complicit.
    ),

    # -------------------------------------------------------------------------
    # CONTACT & OUTGROUP EXPERIENCE
    # -------------------------------------------------------------------------

    (
        "close_friendship_or_relationship_with_outgroup_member",
        "outgroup_affect",
        "up",
        MODERATE,
        # Pettigrew & Tropp (2006) meta-analysis (515 studies): intergroup contact
        # under conditions of equal status, common goals, and institutional support
        # reliably reduces prejudice. Close friendship is the strongest form of contact.
    ),
    (
        "close_friendship_or_relationship_with_outgroup_member",
        "racial_attitudes",
        "up",
        SMALL,
        # Secondary effect: friendship contact shifts attributions for inequality
        # toward structural explanations when the outgroup friend describes
        # discrimination experiences (Pettigrew & Tropp 2006, extended contact).
    ),
    (
        "personal_experience_of_discrimination_or_being_targeted",
        "racial_attitudes",
        "up",
        LARGE,
        # Brewer (1999): direct victimization experience is the strongest
        # individual-level predictor of structural attribution for inequality.
        # Effect persists across racial groups — White respondents who experience
        # class-based discrimination also shift toward structural framing.
    ),
    (
        "personal_experience_of_discrimination_or_being_targeted",
        "institutional_trust",
        "down",
        LARGE,
        # Tyler (2006): experience of unfair treatment by authority figures
        # (police, courts, employers) is the single strongest predictor of
        # institutional legitimacy decline.
    ),
    (
        "living_in_high_diversity_neighborhood_or_community",
        "outgroup_affect",
        "mixed",
        SMALL,
        # Putnam (2007) "E Pluribus Unum": short-term diversity exposure can
        # decrease social trust (hunkering effect), but longitudinal evidence
        # suggests this reverses with sustained positive contact. Mixed direction
        # reflects genuine empirical ambiguity — runtime should resolve by
        # sampling direction probabilistically (60% up, 40% down).
    ),

    # -------------------------------------------------------------------------
    # FAMILY STRUCTURE & INTIMATE LIFE
    # -------------------------------------------------------------------------

    (
        "divorce_or_marital_dissolution",
        "moral_traditionalism",
        "mixed",
        SMALL,
        # Cherlin (2010): divorce produces heterogeneous attitude shifts.
        # Some respondents move toward more permissive views after personal
        # experience; others reaffirm traditional values as a coping response.
        # Net effect is near-zero in aggregate, but individual variance is high.
        # Runtime: sample direction 50/50, draw magnitude from lower SMALL range.
    ),
    (
        "child_or_close_family_member_comes_out_as_lgbtq",
        "moral_traditionalism",
        "down",
        LARGE,
        # Pew (2019): having an LGBTQ family member is the strongest single
        # predictor of attitude liberalization on sexuality items, larger than
        # education or cohort effects. Effect is concentrated in moral_traditionalism
        # and does not strongly generalize to other latent dimensions.
    ),
    (
        "child_or_close_family_member_comes_out_as_lgbtq",
        "outgroup_affect",
        "up",
        SMALL,
        # Secondary generalization effect via Pettigrew & Tropp extended contact.
    ),
    (
        "child_or_family_member_addiction_incarceration_or_crisis",
        "institutional_trust",
        "down",
        MODERATE,
        # McLanahan & Sandefur (1994); Tyler (2006): families navigating the
        # criminal justice or social services system for a member in crisis
        # report significant declines in institutional legitimacy, particularly
        # toward legal and welfare institutions.
    ),
    (
        "child_or_family_member_addiction_incarceration_or_crisis",
        "economic_redistribution",
        "up",
        SMALL,
        # Secondary effect: direct exposure to inadequacy of social safety net
        # increases support for redistribution (Schlozman et al. 2012).
    ),

    # -------------------------------------------------------------------------
    # CIVIC & POLITICAL EXPERIENCE
    # -------------------------------------------------------------------------

    (
        "active_civic_or_political_participation",
        "institutional_trust",
        "up",
        SMALL,
        # Putnam (2000) *Bowling Alone*: civic participation increases bonding
        # and bridging social capital, with modest positive effects on institutional
        # trust. Effect is stronger for local than national institutions.
    ),
    (
        "direct_experience_of_government_failure_or_corruption",
        "institutional_trust",
        "down",
        LARGE,
        # Tyler (2006): direct observation of corruption or incompetence in
        # government produces larger trust declines than media exposure to the
        # same events, consistent with negativity bias in trust updating.
    ),
    (
        "military_service",
        "moral_traditionalism",
        "up",
        MODERATE,
        # Inglehart (1997): institutional socialization in hierarchical,
        # rule-governed environments reinforces traditional value orientations.
        # Effect is moderated by combat exposure — veterans with PTSD show
        # more complex attitude profiles.
    ),
    (
        "military_service",
        "institutional_trust",
        "mixed",
        SMALL,
        # Mixed: military service increases trust in military specifically but
        # can decrease trust in civilian government if service member perceives
        # political misuse of military force (Inglehart 1997).
    ),

]


# -------------------------------------------------------------------------
# RUNTIME HELPERS
# -------------------------------------------------------------------------

def get_updates_for_event(event_type):
    """
    Returns all update rules matching a given life event type.
    Each result is a dict with keys: variable, direction, magnitude_range
    """
    return [
        {
            "variable":        var,
            "direction":       direction,
            "magnitude_range": mag,
        }
        for (ev, var, direction, mag) in DOMAIN_RELEVANCE_MAP
        if ev == event_type
    ]


def list_event_types():
    """Returns deduplicated list of all modeled life event types."""
    return list(dict.fromkeys(ev for (ev, *_) in DOMAIN_RELEVANCE_MAP))


if __name__ == "__main__":
    print(f"Total update rules: {len(DOMAIN_RELEVANCE_MAP)}")
    print(f"Unique event types: {len(list_event_types())}")
    print()
    for ev in list_event_types():
        updates = get_updates_for_event(ev)
        print(f"  {ev}")
        for u in updates:
            print(f"    -> {u['variable']} {u['direction']} {u['magnitude_range']}")
        print()