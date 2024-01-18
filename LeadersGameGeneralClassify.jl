# Load packages
using Random
using Plots
using Distributions

###############
## Constants ##
###############


# Agent Structure 
mutable struct Agent
    id::Int # ID
    value_function::Vector{Float64} # To store the value function for action 1 ("Yield") and action 2 ("Don't Yield")
    choice_history::Vector{String} # To store choice history for data analysis if needed
    temperature::Float64 # softmax_selection temperature
    rewards_against::Dict{Int, Float64} # Total rewards against specific agents
end

function Agent(id::Int)
    return Agent(id, [0.0, 0.0], [], 1.0, Dict{Int, Float64}()) # Initialize agent
end


# Define payoff matrix for Leader game
const PAYOFFS = Dict(
    (1, 1) => (-1.0, -1.0), # Yield, Yield (Cooperator, Cooperator)
    (2, 1) => (2.0, 1.0), # Don't Yield, Yield (Opponent, Cooperator)
    (1, 2) => (1.0, 2.0), # Yield, Don't Yield (Cooperator, Opponent)
    (2, 2) => (-2.0, -2.0) # Don't Yield, Don't Yield (Opponent, Opponent)
)

###############
## Training  ##
###############


#Define a function that updates the value function using the TD(0) formula
function update_value_function!(agent::Agent, action::Int, reward::Float64, alpha::Float64, gamma::Float64)
    max_next_value = maximum(agent.value_function)
    agent.value_function[action] += alpha * (reward + gamma * max_next_value - agent.value_function[action])
end

# Use Softmax to choose actions
function softmax_selection(agent::Agent)
    # Ensure the temperature is positive to avoid division by zero or negative temperatures
    if agent.temperature <= 0
        agent.temperature = 1e-8
    end

    # Calculate the exponentials of the value function divided by temperature
    # and protect against overflow
    max_val = maximum(agent.value_function)
    exp_values = exp.((agent.value_function .- max_val) / agent.temperature)

    # Normalize to get probabilities
    probabilities = exp_values / sum(exp_values)

    # Check if probabilities are valid
    if any(isnan, probabilities) || sum(probabilities) ≈ 0.0
        # Fallback to a uniform distribution or handle the error
        probabilities = fill(1.0 / length(agent.value_function), length(agent.value_function))
    end

    return rand(Categorical(probabilities))
end


function update_temperature!(agent::Agent, iteration::Int, num_rounds::Int, min_temp::Float64)
    # Calculate the decay rate based on the iteration and the total number of rounds
    # This will linearly decrease the temperature from 1.0 to a small positive number by the end of training
    decay_rate = 1 - (1 - min_temp) * (iteration / num_rounds)

    # Apply the decay to the agent's temperature, ensuring it doesn't fall below min_temp
    agent.temperature = max(agent.temperature * decay_rate, min_temp)
end

# TRAINING
function play_game(agent1::Agent, agent2::Agent, alpha::Float64, gamma::Float64)

    action1 = softmax_selection(agent1)
    action2 = softmax_selection(agent2)
    
    reward1, reward2 = PAYOFFS[(action1, action2)]

    # alpha is learning rate and gamma is discount rate
    update_value_function!(agent1, action1, reward1, alpha, gamma)
    update_value_function!(agent2, action2, reward2, alpha, gamma)

    return reward1, reward2
end

#################
### Evaluation ##
#################


# Evaluation Playing 
function play_no_train(agents, n_rounds)
    # Initialize the dictionaries for payoffs and choice histories
    total_rewards = Dict(agent.id => 0.0 for agent in agents)
    cooperator_payoffs = Dict(agent.id => (0.0, 0) for agent in agents)
    opponent_payoffs = Dict(agent.id => (0.0, 0) for agent in agents)

    # Initialize rewards against other agents
    for agent in agents
        agent.rewards_against = Dict((other_agent.id => 0.0 for other_agent in agents if other_agent.id != agent.id))
    end

    for round in 1:n_rounds # Play for n_rounds
        shuffled_agents = shuffle(agents)
        for i in 1:2:length(shuffled_agents)-1
            agent1 = shuffled_agents[i]
            agent2 = shuffled_agents[i+1]

            action1 = softmax_selection(agent1)
            action2 = softmax_selection(agent2)
            
            # Record choices in the agents' history
            push!(agent1.choice_history, action1 == 1 ? "Y" : "NY")
            push!(agent2.choice_history, action2 == 1 ? "Y" : "NY")
            
            reward1, reward2 = PAYOFFS[(action1, action2)]
            
            # Update total rewards
            total_rewards[agent1.id] += reward1
            total_rewards[agent2.id] += reward2
            
            # Update rewards against specific agents
            agent1.rewards_against[agent2.id] += reward1
            agent2.rewards_against[agent1.id] += reward2

           # Update cooperator and opponent payoffs
           if action1 == 1  # Agent1 cooperates
            cooperator_payoffs[agent2.id] = (cooperator_payoffs[agent2.id][1] + reward1, cooperator_payoffs[agent2.id][2] + 1)
        else  # Agent1 does not cooperate
            opponent_payoffs[agent2.id] = (opponent_payoffs[agent2.id][1] + reward1, opponent_payoffs[agent2.id][2] + 1)
        end

        if action2 == 1  # Agent2 cooperates
            cooperator_payoffs[agent1.id] = (cooperator_payoffs[agent1.id][1] + reward2, cooperator_payoffs[agent1.id][2] + 1)
        else  # Agent2 does not cooperate
            opponent_payoffs[agent1.id] = (opponent_payoffs[agent1.id][1] + reward2, opponent_payoffs[agent1.id][2] + 1)
        end
    end
end
    # Return the comprehensive data including total rewards, cooperator payoffs, opponent payoffs, and rewards against each agent
    return total_rewards, cooperator_payoffs, opponent_payoffs
end



function classify_strategy(agent_id, cooperator_payoffs, opponent_payoffs)
    coop_payoff, coop_count = cooperator_payoffs[agent_id]
    opp_payoff, opp_count = opponent_payoffs[agent_id]
    
    # Avoid division by zero in case there are no interactions of a certain type
    avg_coop_payoff = coop_count > 0 ? coop_payoff / coop_count : 0
    avg_opp_payoff = opp_count > 0 ? opp_payoff / opp_count : 0

    println("Agent $agent_id:")
    println("  Cooperator Payoffs: Total = $coop_payoff, Count = $coop_count, Average = $avg_coop_payoff")
    println("  Opponent Payoffs: Total = $opp_payoff, Count = $opp_count, Average = $avg_opp_payoff")
    
    if avg_coop_payoff > avg_opp_payoff
        return "Leader"
    elseif avg_opp_payoff > avg_coop_payoff
        return "Follower"
    else
        return "Equal" # Use this classification if the payoffs are equal
    end
end

function normalize_rewards_per_round(agents, n_rounds)
    num_agents = length(agents)
    normalized_matrix = zeros(num_agents, num_agents)
    
    # Fill in the matrix with normalized rewards per round
    for i in 1:num_agents
        for j in 1:num_agents
            if i != j
                total_reward = agents[i].rewards_against[agents[j].id]
                normalized_matrix[i, j] = total_reward / n_rounds
            end
        end
    end
    
    return normalized_matrix
end


##############
## Plotting ##
##############


# Plot the Value Functions, 
function plot_value_functions(agents)
    p = plot(xlabel="Value for Action 1 (X)", ylabel="Value for Action 2 (Y)", title="Agent Value Functions as Points", legend=:topleft,
             aspect_ratio=:equal)
    for agent in agents
        x_val = agent.value_function[1]
        y_val = agent.value_function[2]
        scatter!(p, [x_val], [y_val], label="Agent $(agent.id)", markersize=8)
    end
    display(p)
end


# Updated plotting function with strategy color-coding
function plot_value_functions_strategy(agents, cooperator_payoffs, opponent_payoffs, min_temp)
    p = plot(xlabel="Value for Yielding", ylabel="Value for Not Yielding", 
             title="Agent Roles, T = $(min_temp)", legend=:topleft, aspect_ratio=:equal, framestyle=:box)

    for agent in agents
        x_val = agent.value_function[1]
        y_val = agent.value_function[2]
        strategy = classify_strategy(agent.id, cooperator_payoffs, opponent_payoffs)
        color = strategy == "Leader" ? :red : :green
        scatter!(p, [x_val], [y_val], label="Agent $(agent.id)", color=color, markersize=8)
    end
    display(p)
    savefig(p, "plots/$(min_temp)_valuefs.png")
end

blue_to_red = cgrad([:blue, :white, :red])
function create_heatmap(normalized_matrix, min_temp)
    h = heatmap(1:size(normalized_matrix,1), 1:size(normalized_matrix,2), normalized_matrix, c=blue_to_red, aspect_ratio=:none, xlabel="Agent ID", ylabel="Agent ID", title="Agent Interaction T = $(min_temp)", xticks=1:size(normalized_matrix,1), yticks=1:size(normalized_matrix,1))
    display(h)
    savefig(h, "plots/$(min_temp)_heatmap.png")
end


####################
##      main      ##
####################

# Function that runs everything
function main(min_temp)
    num_agents = 16
    num_rounds = 100000
    n_test_rounds = 10000
      # This is the minimum temperature we want by the end of training
    alpha = 0.01
    gamma = 0.9
    
    agents = [Agent(i) for i in 1:num_agents]
    
    for round in 1:num_rounds
        shuffled_agents = shuffle(agents)
        for i in 1:2:length(shuffled_agents)-1
            play_game(shuffled_agents[i], shuffled_agents[i+1], alpha, gamma)
        end
        # update temperature
        for agent in agents
            update_temperature!(agent, round, num_rounds, min_temp)
        end
    end

    # plot_value_functions(agents)
    
    # To view the value functions of the agents after the simulation
    for (i, agent) in enumerate(agents)
        println("Agent $i: Value Function = $(agent.value_function)")
        println("$(agent.value_function[2] - agent.value_function[1])")
    end


    # After running play_no_train in main()
    total_rewards, cooperator_payoffs, opponent_payoffs = play_no_train(agents, n_test_rounds)

    plot_value_functions_strategy(agents, cooperator_payoffs, opponent_payoffs, min_temp)


    # Print the last 10 choices of every agent
    for agent in agents
        history_length = length(agent.choice_history)
        last_choices = history_length > 10 ? agent.choice_history[end-9:end] : agent.choice_history
        println("Agent $(agent.id): Last 10 choices: $(last_choices)")
    end

    normalized_matrix = normalize_rewards_per_round(agents, n_test_rounds)
    create_heatmap(normalized_matrix, min_temp)



end

# main(0.1)
for temp in range(0.1, 1, step=0.1)
    main(temp)
end
